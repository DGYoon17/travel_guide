"""写操作确认门与幂等存储。

改造前 `submit_approval` 是一个真实的对外写操作，但 harness 对它没有任何约束：没有确认门、
没有幂等键、不可撤销、不留痕。唯一沾边的是 `PromptStateMachine` 用正则
`(确认|同意|可以|就这样|没问题|提交|发起)` 推进状态——用它来决定是否向 OA 提交一笔上万元
的采购审批，"这个方案可以吗？"就会命中。

现在的机制：
- 模型调用 WRITE 工具时**不执行**，而是登记一个 `PendingAction` 并把摘要交给用户；
- 真正执行只能由用户在独立接口上带 `action_id` 显式触发，不经过任何自然语言判定；
- 执行走幂等键，重复确认返回同一张审批单而不是产生第二张。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class PendingActionStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PendingAction(BaseModel):
    """一次待用户确认的写操作。"""

    action_id: str = Field(default_factory=lambda: f"act_{uuid.uuid4().hex[:16]}")
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    preview: str = ""
    session_id: str
    tenant_id: str
    subject_id: str
    trace_id: str | None = None
    idempotency_key: str
    draft_version: int = 0
    draft_fingerprint: str = ""
    #: 域相关的确认快照（差旅：报价价格与 offer_id；采购：请购单快照）。
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: PendingActionStatus = PendingActionStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
        + timedelta(seconds=settings.confirmation_ttl_seconds)
    )

    @property
    def expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at

    def to_client_payload(self) -> dict[str, Any]:
        """给前端的载荷。不含内部指纹与幂等键。"""
        return {
            "action_id": self.action_id,
            "tool_name": self.tool_name,
            "preview": self.preview,
            "expires_at": self.expires_at.isoformat(),
            "requires_explicit_confirmation": True,
        }


class ConfirmationStore(Protocol):
    async def put(self, action: PendingAction) -> None: ...
    async def get(self, action_id: str, *, tenant_id: str) -> PendingAction | None: ...
    async def delete(self, action_id: str, *, tenant_id: str) -> None: ...


class IdempotencyStore(Protocol):
    async def get(self, key: str, *, tenant_id: str) -> dict[str, Any] | None: ...
    async def put(self, key: str, value: dict[str, Any], *, tenant_id: str) -> None: ...


def _action_key(tenant_id: str, action_id: str) -> str:
    return f"travelmind:pending:{tenant_id}:{action_id}"


def _idem_key(tenant_id: str, key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"travelmind:idem:{tenant_id}:{digest}"


class RedisConfirmationStore:
    def __init__(self, client: Any, *, ttl_seconds: int | None = None) -> None:
        self._client = client
        self._ttl = ttl_seconds or settings.confirmation_ttl_seconds

    async def put(self, action: PendingAction) -> None:
        await self._client.set(
            _action_key(action.tenant_id, action.action_id),
            action.model_dump_json(),
            ex=self._ttl,
        )

    async def get(self, action_id: str, *, tenant_id: str) -> PendingAction | None:
        raw = await self._client.get(_action_key(tenant_id, action_id))
        if not raw:
            return None
        try:
            return PendingAction.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("confirmation.deserialize_failed", action_id=action_id, error=str(exc))
            return None

    async def delete(self, action_id: str, *, tenant_id: str) -> None:
        await self._client.delete(_action_key(tenant_id, action_id))


class RedisIdempotencyStore:
    def __init__(self, client: Any, *, ttl_seconds: int | None = None) -> None:
        self._client = client
        self._ttl = ttl_seconds or settings.idempotency_ttl_seconds

    async def get(self, key: str, *, tenant_id: str) -> dict[str, Any] | None:
        raw = await self._client.get(_idem_key(tenant_id, key))
        if not raw:
            return None
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None

    async def put(self, key: str, value: dict[str, Any], *, tenant_id: str) -> None:
        await self._client.set(
            _idem_key(tenant_id, key),
            json.dumps(value, ensure_ascii=False, default=str),
            ex=self._ttl,
        )


class InMemoryConfirmationStore:
    def __init__(self) -> None:
        self._data: dict[str, PendingAction] = {}

    async def put(self, action: PendingAction) -> None:
        self._data[_action_key(action.tenant_id, action.action_id)] = action

    async def get(self, action_id: str, *, tenant_id: str) -> PendingAction | None:
        return self._data.get(_action_key(tenant_id, action_id))

    async def delete(self, action_id: str, *, tenant_id: str) -> None:
        self._data.pop(_action_key(tenant_id, action_id), None)


class InMemoryIdempotencyStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    async def get(self, key: str, *, tenant_id: str) -> dict[str, Any] | None:
        return self._data.get(_idem_key(tenant_id, key))

    async def put(self, key: str, value: dict[str, Any], *, tenant_id: str) -> None:
        self._data[_idem_key(tenant_id, key)] = value


def build_confirmation_store(redis_client: Any | None) -> ConfirmationStore:
    if redis_client is None:
        logger.warning("confirmation.store_degraded_to_memory")
        return InMemoryConfirmationStore()
    return RedisConfirmationStore(redis_client)


def build_idempotency_store(redis_client: Any | None) -> IdempotencyStore:
    if redis_client is None:
        logger.warning("idempotency.store_degraded_to_memory")
        return InMemoryIdempotencyStore()
    return RedisIdempotencyStore(redis_client)
