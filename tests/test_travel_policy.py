"""差旅制度引擎测试：差标、超标金额、审批链、硬约束。"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.domain.travel.models import (
    CabinClass,
    TravelerLevel,
    TripPurpose,
    TripRequest,
    TripType,
)
from app.domain.travel.policy import default_travel_policy, evaluate_trip


TODAY = date(2026, 11, 11)


def make_trip(**overrides) -> TripRequest:
    payload = {
        "trip_id": "trip_1",
        "traveler_id": "E10086",
        "traveler_level": TravelerLevel.MANAGER,
        "department": "市场部",
        "origin_city": "上海",
        "destination_city": "北京",
        "depart_date": date(2026, 12, 10),
        "trip_type": TripType.ONE_WAY,
        "purpose": TripPurpose.CLIENT,
        "cabin_class": CabinClass.ECONOMY,
        "hotel_needed": False,
        "budget_cny": Decimal("5000"),
    }
    payload.update(overrides)
    return TripRequest(**payload)


def test_hotel_over_standard_reports_nightly_delta() -> None:
    policy = default_travel_policy(version="test-v1")
    # 北京 = 一线城市，经理住宿上限 800/晚；住 2 晚、1000/晚 → 超标 400。
    trip = make_trip(
        destination_city="北京",
        depart_date=date(2026, 12, 10),
        return_date=date(2026, 12, 12),
        trip_type=TripType.ROUND_TRIP,
        hotel_needed=True,
    )
    result = evaluate_trip(
        policy, trip, hotel_nightly_cny=Decimal("1000"), total_cny=Decimal("4200"), today=TODAY
    )
    finding = next(f for f in result.findings if f.code == "HOTEL_OVER_STANDARD")
    assert finding.over_standard_cny == Decimal("400")
    assert result.over_standard_total_cny == Decimal("400")
    # 超标是"填理由 + 加签"，不是硬拒绝。
    assert result.compliant is True
    assert "费用合规复核" in result.required_roles


def test_cabin_over_standard_for_manager_domestic() -> None:
    policy = default_travel_policy(version="test-v1")
    trip = make_trip(cabin_class=CabinClass.BUSINESS)
    result = evaluate_trip(policy, trip, today=TODAY)
    codes = {f.code for f in result.findings}
    assert "CABIN_OVER_STANDARD" in codes
    assert "分管高管" in result.required_roles


def test_international_without_passport_blocks() -> None:
    policy = default_travel_policy(version="test-v1")
    trip = make_trip(
        destination_city="东京",
        international=True,
        passport_ready=False,
        depart_date=date(2026, 12, 20),
        estimated_flight_minutes=180,
    )
    result = evaluate_trip(policy, trip, today=TODAY)
    assert result.compliant is False
    assert any(f.code == "PASSPORT_MISSING" for f in result.blocking)
    assert "签证专员" in result.required_roles


def test_advance_days_warns_but_does_not_block() -> None:
    policy = default_travel_policy(version="test-v1")
    trip = make_trip(depart_date=date(2026, 11, 12))  # 距今天 1 天 < 国内 3 天
    result = evaluate_trip(policy, trip, today=TODAY)
    codes = {f.code for f in result.findings}
    assert "ADVANCE_DAYS" in codes
    assert result.compliant is True


def test_high_amount_adds_finance_and_executive() -> None:
    policy = default_travel_policy(version="test-v1")
    trip = make_trip()
    result = evaluate_trip(policy, trip, total_cny=Decimal("25000"), today=TODAY)
    assert result.required_roles[:2] == ["部门负责人", "差旅管理员"]
    assert "财务BP" in result.required_roles
    assert "分管高管" in result.required_roles


def test_restricted_destination_blocks() -> None:
    policy = default_travel_policy(version="test-v1")
    trip = make_trip(destination_city="叙利亚", international=True)
    result = evaluate_trip(policy, trip, today=TODAY)
    assert result.compliant is False
    assert any(f.code == "RESTRICTED_DESTINATION" for f in result.blocking)


def test_round_trip_requires_return_date() -> None:
    policy = default_travel_policy(version="test-v1")
    trip = make_trip(trip_type=TripType.ROUND_TRIP, return_date=None)
    result = evaluate_trip(policy, trip, today=TODAY)
    assert any(f.code == "RETURN_DATE_REQUIRED" for f in result.blocking)


def test_policy_version_is_reported() -> None:
    policy = default_travel_policy(version="2026.11-v9")
    result = evaluate_trip(policy, make_trip(), today=TODAY)
    assert result.policy_version == "2026.11-v9"
