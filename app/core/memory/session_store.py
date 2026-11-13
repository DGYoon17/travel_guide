"""L1 + L2 会话状态的持久化。

改造前 `session_id` 只是被原样回填进响应，从未用于加载或写入：多轮上下文完全靠前端每次
重传全量 messages。这在采购场景意味着"用户已确认的金额和审批人"由客户端持有并可篡改。

现在会话状态存服务端（Redis），前端只需传 `session_id`。Redis 不可用时退化到进程内存并
显式告警——退化后多副本不共享，但不会静默丢状态。

并发写用版本号做乐观锁：同一会话的两个并发请求不会互相覆盖对方的槽位确认。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.config import settings
from app.core.logging import get_logger
from app.core.memory.draft import RequisitionDraft
from app.core.prompt.state_machine import ConversationState

logger = get_logger(__name__)


class SessionState(BaseModel):
    """一次会话的完整服务端状态。"""

    session_id: str
    tenant_id: str = "default"
    subject_id: str = ""
    version: int = 0
    turn: int = 0
    conversation_state: ConversationState = ConversationState.GREETING
    #: `ShortTermMemory.dump()` 的结果：摘要 + 消息列表。
    memory: dict[str, Any] = Field(default_factory=dict)
    draft: RequisitionDraft = Field(default_factory=RequisitionDraft)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)


class ConcurrentSessionUpdate(RuntimeError):
    """乐观锁冲突：会话在本次读取之后被其它请求改过。"""


class SessionStore(Protocol):
    async def load(self, session_id: str, *, tenant_id: str) -> SessionState | None: ...
    async def save(self, state: SessionState) -> SessionState: ...
    async def delete(self, session_id: str, *, tenant_id: str) -> None: ...


def _key(tenant_id: str, session_id: str) -> str:
    # 租户前缀是硬隔离的一部分：不同法人的同名 session_id 不能互相读到。
    return f"procuremind:sess:{tenant_id}:{session_id}"


class RedisSessionStore:
    """Redis 后端。使用 WATCH/MULTI 做版本号 CAS。"""

    def __init__(self, client: Any, *, ttl_seconds: int | None = None) -> None:
        self._client = client
        self._ttl = ttl_seconds or settings.session_ttl_seconds

    async def load(self, session_id: str, *, tenant_id: str) -> SessionState | None:
        raw = await self._client.get(_key(tenant_id, session_id))
        if not raw:
            return None
        try:
            return SessionState.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("session.deserialize_failed", session_id=session_id, error=str(exc))
            return None

    async def save(self, state: SessionState) -> SessionState:
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
                    # redis.WatchError 等并发异常：重试一次读-改-写。
                    if type(exc).__name__ == "WatchError":
                        continue
                    raise

    async def delete(self, session_id: str, *, tenant_id: str) -> None:
        await self._client.delete(_key(tenant_id, session_id))


class InMemorySessionStore:
    """进程内后端。用于测试，以及 Redis 不可用时的显式降级。"""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def load(self, session_id: str, *, tenant_id: str) -> SessionState | None:
        raw = self._data.get(_key(tenant_id, session_id))
        return SessionState.model_validate_json(raw) if raw else None

    async def save(self, state: SessionState) -> SessionState:
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


def build_session_store(redis_client: Any | None) -> SessionStore:
    if redis_client is None:
        logger.warning("session.store_degraded_to_memory", reason="redis unavailable")
        return InMemorySessionStore()
    return RedisSessionStore(redis_client)
