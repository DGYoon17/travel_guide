"""L1 + L2 差旅会话状态的持久化。

与采购版共用同一套语义（Redis CAS、租户前缀、主体绑定），但草稿类型换成 `TripDraft`。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.config import settings
from app.core.logging import get_logger
from app.core.memory.trip_draft import TripDraft
from app.core.prompt.travel_state import TravelState

logger = get_logger(__name__)


class TravelSessionState(BaseModel):
    session_id: str
    tenant_id: str = "default"
    subject_id: str = ""
    version: int = 0
    turn: int = 0
    conversation_state: TravelState = TravelState.GREETING
    memory: dict[str, Any] = Field(default_factory=dict)
    draft: TripDraft = Field(default_factory=TripDraft)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)


class ConcurrentSessionUpdate(RuntimeError):
    """乐观锁冲突。"""


class TravelSessionStore(Protocol):
    async def load(self, session_id: str, *, tenant_id: str) -> TravelSessionState | None: ...
    async def save(self, state: TravelSessionState) -> TravelSessionState: ...
    async def delete(self, session_id: str, *, tenant_id: str) -> None: ...


def _key(tenant_id: str, session_id: str) -> str:
    return f"travelmind:sess:{tenant_id}:{session_id}"


class RedisTravelSessionStore:
    def __init__(self, client: Any, *, ttl_seconds: int | None = None) -> None:
        self._client = client
        self._ttl = ttl_seconds or settings.session_ttl_seconds

    async def load(self, session_id: str, *, tenant_id: str) -> TravelSessionState | None:
        raw = await self._client.get(_key(tenant_id, session_id))
        if not raw:
            return None
        try:
            return TravelSessionState.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("session.deserialize_failed", session_id=session_id, error=str(exc))
            return None

    async def save(self, state: TravelSessionState) -> TravelSessionState:
        key = _key(state.tenant_id, state.session_id)
        expected = state.version
        state.version = expected + 1
        state.touch()
        payload = state.model_dump_json()

        async with self._client.pipeline() as pipe:
            while True:
                try:
                    await pipe.watch(key)
                    current = await pipe.get(key)
                    if current:
                        try:
                            current_version = int(json.loads(current).get("version", 0))
                        except (json.JSONDecodeError, TypeError, ValueError):
                            current_version = expected
                        if current_version != expected:
                            await pipe.unwatch()
                            raise ConcurrentSessionUpdate(
                                f"会话 {state.session_id} 已被其它请求更新"
                                f"（期望版本 {expected}，实际 {current_version}）"
                            )
                    pipe.multi()
                    await pipe.set(key, payload, ex=self._ttl)
                    await pipe.execute()
                    return state
                except ConcurrentSessionUpdate:
                    raise
                except Exception as exc:  # noqa: BLE001
                    if type(exc).__name__ == "WatchError":
                        continue
                    raise

    async def delete(self, session_id: str, *, tenant_id: str) -> None:
        await self._client.delete(_key(tenant_id, session_id))


class InMemoryTravelSessionStore:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def load(self, session_id: str, *, tenant_id: str) -> TravelSessionState | None:
        raw = self._data.get(_key(tenant_id, session_id))
        return TravelSessionState.model_validate_json(raw) if raw else None

    async def save(self, state: TravelSessionState) -> TravelSessionState:
        key = _key(state.tenant_id, state.session_id)
        existing = self._data.get(key)
        if existing:
            try:
                current_version = int(json.loads(existing).get("version", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                current_version = state.version
            if current_version != state.version:
                raise ConcurrentSessionUpdate(
                    f"会话 {state.session_id} 已被其它请求更新"
                    f"（期望版本 {state.version}，实际 {current_version}）"
                )
        state.version += 1
        state.touch()
        self._data[key] = state.model_dump_json()
        return state

    async def delete(self, session_id: str, *, tenant_id: str) -> None:
        self._data.pop(_key(tenant_id, session_id), None)


def build_travel_session_store(redis_client: Any | None) -> TravelSessionStore:
    if redis_client is None:
        logger.warning("session.store_degraded_to_memory", reason="redis unavailable")
        return InMemoryTravelSessionStore()
    return RedisTravelSessionStore(redis_client)


__all__ = [
    "ConcurrentSessionUpdate",
    "InMemoryTravelSessionStore",
    "RedisTravelSessionStore",
    "TravelSessionState",
    "TravelSessionStore",
    "build_travel_session_store",
]
