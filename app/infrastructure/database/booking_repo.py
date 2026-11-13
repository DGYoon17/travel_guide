"""预订结果的 PostgreSQL 持久化。

审计表记录"谁确认了什么"，订单表记录"实际订到了什么"。两者职责不同：
审计 append-only，订单需要按 `booking_id` 幂等 upsert（状态会从 confirmed 变为 cancelled）。

金额用字符串存，避免不同数据库/驱动的 Decimal 精度差异；展示与对账以供应方订单为准。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Index, MetaData, String, Table, Text, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.logging import get_logger
from app.domain.travel.models import BookingRecord

logger = get_logger(__name__)

metadata = MetaData()

travel_bookings = Table(
    "travel_bookings",
    metadata,
    Column("booking_id", String(64), primary_key=True),
    Column("tenant_id", String(64), nullable=False),
    Column("subject_id", String(64), nullable=False),
    Column("trip_id", String(64), nullable=False),
    Column("offer_id", String(64), nullable=False),
    Column("kind", String(16), nullable=False),
    Column("status", String(16), nullable=False),
    Column("supplier", String(128), nullable=False),
    Column("title", String(512), nullable=False),
    Column("price_cny", String(32), nullable=False),
    Column("order_no", String(64), nullable=False),
    Column("idempotency_key", String(256), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("note", Text, nullable=False, server_default=""),
    Index("ix_booking_tenant_subject", "tenant_id", "subject_id"),
    Index("ix_booking_trip", "trip_id"),
)


class SqlBookingRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def ensure_schema(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def save(self, record: BookingRecord) -> None:
        values: dict[str, Any] = {
            "booking_id": record.booking_id,
            "tenant_id": record.tenant_id or "default",
            "subject_id": record.traveler_id,
            "trip_id": record.trip_id,
            "offer_id": record.offer_id,
            "kind": record.kind.value,
            "status": record.status.value,
            "supplier": record.supplier,
            "title": record.title[:512],
            "price_cny": str(record.price_cny),
            "order_no": record.order_no,
            "idempotency_key": record.idempotency_key,
            "created_at": record.created_at,
            "note": record.note,
        }
        statement = pg_insert(travel_bookings).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=["booking_id"],
            set_={
                "status": statement.excluded.status,
                "note": statement.excluded.note,
                "price_cny": statement.excluded.price_cny,
            },
        )
        async with self._engine.begin() as conn:
            await conn.execute(statement)

    async def list_by_subject(
        self, *, tenant_id: str, subject_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        statement = (
            select(travel_bookings)
            .where(
                travel_bookings.c.tenant_id == tenant_id,
                travel_bookings.c.subject_id == subject_id,
            )
            .order_by(travel_bookings.c.created_at.desc())
            .limit(limit)
        )
        async with self._engine.begin() as conn:
            rows = (await conn.execute(statement)).mappings().all()
        return [dict(row) for row in rows]


__all__ = ["SqlBookingRepository", "travel_bookings"]
