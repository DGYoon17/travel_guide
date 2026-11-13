"""L5 事实日志：不可变的决策留痕。

"用户在 T 时刻确认了总额 18800、供应商 X、同意提交给财务BP" 这类事实不能存在于会被摘要
压缩、被窗口裁剪的对话记忆里。它是审计对象，也是幂等与撤销的依据，因此单独 append-only
存储，只写不改。

后端可替换：有 Postgres 时用 `SqlAuditLog`，否则退化到有界内存实现并打警告——退化是显式
的，不会静默丢审计。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger(__name__)


class AuditAction(str, Enum):
    TOOL_INVOKED = "tool.invoked"
    TOOL_FAILED = "tool.failed"
    ACTION_PENDING = "action.pending"
    ACTION_CONFIRMED = "action.confirmed"
    ACTION_REJECTED = "action.rejected"
    ACTION_REPLAYED = "action.replayed"
    APPROVAL_SUBMITTED = "approval.submitted"
    DRAFT_CONFIRMED = "draft.confirmed"
    KNOWLEDGE_INGESTED = "knowledge.ingested"
    KNOWLEDGE_PUBLISHED = "knowledge.published"
    MEMORY_WRITTEN = "memory.written"
    DEGRADED_RESPONSE = "response.degraded"


#: 审计载荷里需要掩码的键（片段匹配）。
_SENSITIVE_KEY_PARTS: tuple[str, ...] = (
    "token", "secret", "password", "api_key", "apikey",
    "authorization", "bank_account", "id_card", "phone", "mobile",
)


def redact(payload: Any, *, max_chars: int = 4000) -> Any:
    """掩码敏感键并限制体积，避免把凭据和大段正文写进审计表。"""
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for key, value in payload.items():
            lowered = str(key).lower()
            if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
                out[key] = "***"
            else:
                out[key] = redact(value, max_chars=max_chars)
        return out
    if isinstance(payload, (list, tuple)):
        return [redact(v, max_chars=max_chars) for v in payload][:50]
    if isinstance(payload, str) and len(payload) > max_chars:
        return payload[:max_chars] + f"…(截断，原长度 {len(payload)})"
    return payload


def payload_hash(payload: Any) -> str:
    """载荷内容哈希。留痕时可只存哈希以避免落敏感原文。"""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AuditEvent(BaseModel):
    """一条不可变事实。"""

    event_id: str = Field(default_factory=lambda: f"aud_{uuid.uuid4().hex}")
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tenant_id: str = "default"
    actor: str = "system"
    session_id: str | None = None
    trace_id: str | None = None
    action: AuditAction
    resource_type: str = ""
    resource_id: str | None = None
    outcome: str = "ok"
    policy_version: str | None = None
    prompt_version: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_sha256: str = ""

    def finalized(self) -> "AuditEvent":
        safe = redact(self.payload)
        return self.model_copy(
            update={"payload": safe, "payload_sha256": self.payload_sha256 or payload_hash(self.payload)}
        )


class AuditLog(Protocol):
    async def record(self, event: AuditEvent) -> None: ...
    async def recent(self, *, tenant_id: str | None = None, limit: int = 50) -> list[AuditEvent]: ...


class InMemoryAuditLog:
    """有界内存实现。用于测试和「数据库不可用」的显式降级。"""

    def __init__(self, maxlen: int = 5000) -> None:
        self._events: deque[AuditEvent] = deque(maxlen=maxlen)

    async def record(self, event: AuditEvent) -> None:
        final = event.finalized()
        self._events.append(final)
        logger.info(
            "audit.event",
            action=final.action.value,
            actor=final.actor,
            resource=f"{final.resource_type}:{final.resource_id}",
            outcome=final.outcome,
            trace_id=final.trace_id,
            durable=False,
        )

    async def recent(self, *, tenant_id: str | None = None, limit: int = 50) -> list[AuditEvent]:
        items = [e for e in self._events if tenant_id is None or e.tenant_id == tenant_id]
        return items[-limit:][::-1]


class NullAuditLog:
    """审计关闭时的空实现（仅非生产环境允许）。"""

    async def record(self, event: AuditEvent) -> None:
        return None

    async def recent(self, *, tenant_id: str | None = None, limit: int = 50) -> list[AuditEvent]:
        return []


async def safe_record(audit: AuditLog | None, event: AuditEvent) -> None:
    """审计失败不应该让业务请求失败，但必须留下告警。"""
    if audit is None:
        return
    try:
        await audit.record(event)
    except Exception as exc:  # noqa: BLE001
        logger.error("audit.record_failed", action=event.action.value, error=str(exc))
