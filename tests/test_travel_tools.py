"""差旅工具测试：槽位回写、方案生成、差标校验、预订与退订。"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.core.memory.profile import ClaimsProfileProvider
from app.core.memory.trip_draft import TripDraft
from app.core.tools.errors import ToolErrorCode, ToolFailure, ToolSuccess
from app.core.tools.spec import ToolContext
from app.core.tools.travel_builtin import build_travel_registry
from app.domain.travel.policy import default_travel_policy
from app.infrastructure.suppliers.mock_travel import MockTravelSupplier
from tests.conftest import COMPLETE_TRIP_SLOTS, make_travel_principal


async def make_ctx(**slot_overrides) -> tuple[ToolContext, MockTravelSupplier]:
    principal = make_travel_principal()
    profile = await ClaimsProfileProvider().get(principal)
    assert profile is not None
    draft = TripDraft(session_id="s1", tenant_id=principal.tenant_id, subject_id=principal.subject_id)
    slots = dict(COMPLETE_TRIP_SLOTS)
    slots.update(slot_overrides)
    from app.core.memory.slots import SlotSource

    draft.merge(slots, source=SlotSource.USER, turn=1)
    supplier = MockTravelSupplier()
    ctx = ToolContext(
        principal=principal,
        profile=profile,
        draft=draft,
        policy=default_travel_policy(version="test-v1"),
        session_id="s1",
        trace_id="t1",
        today=date(2026, 11, 11),
        supplier=supplier,
    )
    return ctx, supplier


async def test_plan_trip_writes_slots_and_builds_plan() -> None:
    ctx, _ = await make_ctx()
    # 即使草稿为空，工具参数也能回写 L2 后再规划——避免模型说了日期、草稿里没有。
    ctx.draft = TripDraft(session_id="s1")
    registry = build_travel_registry()
    result = await registry.invoke(
        "plan_trip",
        {
            "origin_city": "上海",
            "destination_city": "北京",
            "depart_date": "2027-03-10",
            "purpose": "client",
            "cabin_class": "economy",
        },
        ctx,
    )
    assert isinstance(result, ToolSuccess), result
    plan = ctx.draft.last_plan
    assert plan is not None
    assert plan["plan_id"].startswith("plan_")
    assert plan["segments"]
    assert Decimal(plan["total_cny"]) > 0
    assert ctx.draft.value("destination_city") == "北京"
    assert plan["approval_path"][0] == "部门负责人"


async def test_search_flights_updates_route_slots() -> None:
    ctx, _ = await make_ctx()
    ctx.draft = TripDraft(session_id="s1")
    registry = build_travel_registry()
    result = await registry.invoke(
        "search_flights",
        {
            "origin_city": "上海",
            "destination_city": "深圳",
            "depart_date": "2027-03-10",
        },
        ctx,
    )
    assert isinstance(result, ToolSuccess), result
    assert ctx.draft.value("origin_city") == "上海"
    assert ctx.draft.value("destination_city") == "深圳"
    assert ctx.draft.value("depart_date") == "2027-03-10"


async def test_check_compliance_flags_hotel_over_standard() -> None:
    ctx, _ = await make_ctx(hotel_needed=True, trip_type="round_trip", return_date="2027-03-12")
    registry = build_travel_registry()
    result = await registry.invoke("check_travel_compliance", {"hotel_nightly_cny": "1200"}, ctx)
    assert isinstance(result, ToolSuccess), result
    codes = {f["code"] for f in result.data["compliance"]["findings"]}
    assert "HOTEL_OVER_STANDARD" in codes
    assert Decimal(result.data["compliance"]["over_standard_total_cny"]) > 0


async def test_book_trip_requires_plan() -> None:
    ctx, _ = await make_ctx()
    registry = build_travel_registry()
    result = await registry.invoke("book_trip", {}, ctx)
    assert isinstance(result, ToolFailure)
    assert result.code is ToolErrorCode.PRECONDITION_NOT_MET


async def test_book_trip_creates_bookings_and_cancel_works() -> None:
    ctx, supplier = await make_ctx()
    registry = build_travel_registry()
    planned = await registry.invoke("plan_trip", {}, ctx)
    assert isinstance(planned, ToolSuccess), planned

    booked = await registry.invoke("book_trip", {}, ctx)
    assert isinstance(booked, ToolSuccess), booked
    booking_ids = ctx.draft.booking_ids
    assert len(booking_ids) == len(ctx.draft.last_plan["segments"])

    cancelled = await registry.invoke(
        "cancel_booking", {"booking_id": booking_ids[0], "reason": "行程取消"}, ctx
    )
    assert isinstance(cancelled, ToolSuccess), cancelled
    assert booking_ids[0] not in ctx.draft.booking_ids
    record = await supplier.get_booking(booking_ids[0])
    assert record is not None
    assert record.status.value == "cancelled"


async def test_book_trip_persists_booking_records() -> None:
    class RecordingRepo:
        def __init__(self) -> None:
            self.saved = []

        async def save(self, record) -> None:
            self.saved.append(record)

    ctx, _ = await make_ctx()
    repo = RecordingRepo()
    ctx.booking_repo = repo
    registry = build_travel_registry()
    planned = await registry.invoke("plan_trip", {}, ctx)
    assert isinstance(planned, ToolSuccess), planned
    booked = await registry.invoke("book_trip", {}, ctx)
    assert isinstance(booked, ToolSuccess), booked
    assert len(repo.saved) == len(ctx.draft.last_plan["segments"])
    assert all(record.tenant_id == "acme" for record in repo.saved)


async def test_submit_approval_after_plan() -> None:
    ctx, _ = await make_ctx()
    registry = build_travel_registry()
    planned = await registry.invoke("plan_trip", {}, ctx)
    assert isinstance(planned, ToolSuccess), planned
    result = await registry.invoke("submit_travel_approval", {"note": "客户拜访"}, ctx)
    assert isinstance(result, ToolSuccess), result
    assert ctx.draft.submitted_approval_no
    assert result.data["approval"]["approval_no"] == ctx.draft.submitted_approval_no


async def test_hotel_date_order_is_field_level_error() -> None:
    ctx, _ = await make_ctx()
    registry = build_travel_registry()
    result = await registry.invoke(
        "search_hotels",
        {"city": "北京", "check_in": "2027-03-10", "check_out": "2027-03-09"},
        ctx,
    )
    assert isinstance(result, ToolFailure)
    assert result.code is ToolErrorCode.INVALID_ARGUMENTS
    assert result.field_issues[0].field == "check_out"


async def test_plan_trip_blocks_without_destination() -> None:
    ctx, _ = await make_ctx()
    ctx.draft = TripDraft(session_id="s1")
    registry = build_travel_registry()
    result = await registry.invoke("plan_trip", {}, ctx)
    assert isinstance(result, ToolFailure)
    assert result.code is ToolErrorCode.MISSING_ARGUMENTS
    assert "destination_city" in result.details.get("missing_fields", [])


async def test_supplier_price_change_is_visible_in_quote() -> None:
    ctx, supplier = await make_ctx()
    registry = build_travel_registry()
    planned = await registry.invoke("plan_trip", {}, ctx)
    assert isinstance(planned, ToolSuccess), planned
    offer_id = ctx.draft.last_plan["segments"][0]["offer_id"]
    supplier.simulate_price_change(offer_id, price_cny=Decimal("9999"))
    quote = await supplier.quote(offer_id)
    assert quote.price_cny == Decimal("9999.00")
