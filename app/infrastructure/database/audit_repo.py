"""L5 事实日志的 Postgres 后端（append-only）。

只暴露 insert 与 select，不提供 update/delete —— 审计表的不可变性靠"没有写路径"保证，
而不是靠调用方自律。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    MetaData,
    String,
    Table,
    Text,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.logging import get_logger
from app.core.memory.audit import AuditAction, AuditEvent

logger = get_logger(__name__)

metadata = MetaData()

audit_events = Table(
    "audit_events",
    metadata,
    Column("event_id", String(64), primary_key=True),
    Column("occurred_at", DateTime(timezone=True), nullable=False, index=True),
    Column("tenant_id", String(64), nullable=False),
    Column("actor", String(128), nullable=False),
    Column("session_id", String(128), nullable=True),
    Column("trace_id", String(64), nullable=True),
    Column("action", String(64), nullable=False),
    Column("resource_type", String(64), nullable=False, server_default=""),
    Column("resource_id", String(128), nullable=True),
    Column("outcome", String(32), nullable=False, server_default="ok"),
    Column("policy_version", String(64), nullable=True),
    Column("prompt_version", String(64), nullable=True),
    Column("payload", JSONB, nullable=False),
    Column("payload_sha256", Text, nullable=False),
    Index("ix_audit_tenant_time", "tenant_id", "occurred_at"),
    Index("ix_audit_resource", "resource_type", "resource_id"),
)


class SqlAuditLog:
    """基于 SQLAlchemy Core 的审计写入器。"""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def ensure_schema(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
        logger.info("audit.schema_ready", table="audit_events")

    async def record(self, event: AuditEvent) -> None:
        final = event.finalized()
        async with self._engine.begin() as conn:
            await conn.execute(
                audit_events.insert().values(
                    event_id=final.event_id,
                    occurred_at=final.occurred_at,
                    tenant_id=final.tenant_id,
                    actor=final.actor,
                    session_id=final.session_id,
                    trace_id=final.trace_id,
                    action=final.action.value,
                    resource_type=final.resource_type,
                    resource_id=final.resource_id,
                    outcome=final.outcome,
                    policy_version=final.policy_version,
                    prompt_version=final.prompt_version,
                    payload=final.payload,
                    payload_sha256=final.payload_sha256,
                )
            )

    async def recent(self, *, tenant_id: str | None = None, limit: int = 50) -> list[AuditEvent]:
        stmt = select(audit_events).order_by(audit_events.c.occurred_at.desc()).limit(limit)
        if tenant_id is not None:
            stmt = stmt.where(audit_events.c.tenant_id == tenant_id)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()
        return [_row_to_event(dict(r)) for r in rows]


def _row_to_event(row: dict[str, Any]) -> AuditEvent:
    occurred = row.get("occurred_at")
    return AuditEvent(
        event_id=str(row["event_id"]),
        occurred_at=occurred if isinstance(occurred, datetime) else datetime.now().astimezone(),
        tenant_id=str(row.get("tenant_id") or "default"),
        actor=str(row.get("actor") or "system"),
        session_id=row.get("session_id"),
        trace_id=row.get("trace_id"),
        action=AuditAction(row["action"]),
        resource_type=str(row.get("resource_type") or ""),
        resource_id=row.get("resource_id"),
        outcome=str(row.get("outcome") or "ok"),
        policy_version=row.get("policy_version"),
        prompt_version=row.get("prompt_version"),
        payload=dict(row.get("payload") or {}),
        payload_sha256=str(row.get("payload_sha256") or ""),
    )
