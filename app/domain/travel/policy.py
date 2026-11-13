"""差旅制度引擎：确定性规则，不经过 LLM。

三条原则：

1. **标准可解释**：每条结论都带 `code` 与 `expected`，模型只能转述，不能自行发明差标；
2. **超标不等于拒绝**：舱位/住宿超标通常允许"填理由 + 加签"，只有证件缺失、目的地受限
   这类硬约束才是 `block`；
3. **审批链由金额与风险共同决定**：金额决定财务/高管节点，国际行程与超标决定合规节点。
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.domain.travel.models import (
    CABIN_LABELS,
    CABIN_RANK,
    CITY_TIER_LABELS,
    CabinClass,
    CityTier,
    ComplianceResult,
    PolicyFinding,
    TravelerLevel,
    TravelPolicy,
    TripRequest,
    TripType,
    infer_city_tier,
)


def default_travel_policy(company_id: str = "default", *, version: str = "2026.11-v1") -> TravelPolicy:
    return TravelPolicy(
        policy_id=f"TRAVEL-{company_id}",
        company_id=company_id,
        version=version,
        effective_from=date(2026, 1, 1),
        domestic_cabin={
            TravelerLevel.STAFF: CabinClass.ECONOMY,
            TravelerLevel.MANAGER: CabinClass.ECONOMY,
            TravelerLevel.DIRECTOR: CabinClass.PREMIUM_ECONOMY,
            TravelerLevel.EXECUTIVE: CabinClass.BUSINESS,
        },
        international_cabin={
            TravelerLevel.STAFF: CabinClass.PREMIUM_ECONOMY,
            TravelerLevel.MANAGER: CabinClass.PREMIUM_ECONOMY,
            TravelerLevel.DIRECTOR: CabinClass.BUSINESS,
            TravelerLevel.EXECUTIVE: CabinClass.BUSINESS,
        },
        long_haul_minutes=360,
        long_haul_cabin={
            TravelerLevel.STAFF: CabinClass.PREMIUM_ECONOMY,
            TravelerLevel.MANAGER: CabinClass.PREMIUM_ECONOMY,
            TravelerLevel.DIRECTOR: CabinClass.BUSINESS,
            TravelerLevel.EXECUTIVE: CabinClass.BUSINESS,
        },
        hotel_caps={
            CityTier.TIER_1: {
                TravelerLevel.STAFF: Decimal("600"),
                TravelerLevel.MANAGER: Decimal("800"),
                TravelerLevel.DIRECTOR: Decimal("1200"),
                TravelerLevel.EXECUTIVE: Decimal("1800"),
            },
            CityTier.NEW_TIER_1: {
                TravelerLevel.STAFF: Decimal("500"),
                TravelerLevel.MANAGER: Decimal("700"),
                TravelerLevel.DIRECTOR: Decimal("1000"),
                TravelerLevel.EXECUTIVE: Decimal("1500"),
            },
            CityTier.OTHER: {
                TravelerLevel.STAFF: Decimal("400"),
                TravelerLevel.MANAGER: Decimal("550"),
                TravelerLevel.DIRECTOR: Decimal("800"),
                TravelerLevel.EXECUTIVE: Decimal("1200"),
            },
        },
        advance_days_domestic=3,
        advance_days_international=7,
        finance_review_above_cny={
            TravelerLevel.STAFF: Decimal("3000"),
            TravelerLevel.MANAGER: Decimal("6000"),
            TravelerLevel.DIRECTOR: Decimal("12000"),
            TravelerLevel.EXECUTIVE: Decimal("25000"),
        },
        executive_review_above_cny={
            TravelerLevel.STAFF: Decimal("10000"),
            TravelerLevel.MANAGER: Decimal("20000"),
            TravelerLevel.DIRECTOR: Decimal("40000"),
            TravelerLevel.EXECUTIVE: Decimal("80000"),
        },
        restricted_destinations=frozenset({"叙利亚", "阿富汗", "也门", "索马里", "利比亚"}),
        international_requires_compliance=True,
        receipt_required_above_cny=Decimal("500"),
        over_standard_reason_required=True,
    )


def allowed_cabin(policy: TravelPolicy, trip: TripRequest) -> CabinClass:
    """按职级、国际/国内、飞行时长计算可订的最高舱位。"""
    level = trip.traveler_level
    if trip.international:
        allowed = policy.international_cabin.get(level, CabinClass.ECONOMY)
    else:
        allowed = policy.domestic_cabin.get(level, CabinClass.ECONOMY)

    if trip.estimated_flight_minutes >= policy.long_haul_minutes:
        long_haul = policy.long_haul_cabin.get(level)
        if long_haul and CABIN_RANK[long_haul] > CABIN_RANK[allowed]:
            allowed = long_haul
    return allowed


def evaluate_trip(
    policy: TravelPolicy,
    trip: TripRequest,
    *,
    hotel_nightly_cny: Decimal | None = None,
    total_cny: Decimal | None = None,
    today: date | None = None,
) -> ComplianceResult:
    """对一次行程做完整差标校验，并给出审批链。"""
    today = today or date.today()
    findings: list[PolicyFinding] = []
    over_standard = Decimal("0")

    # ------------------------------------------------------------------ 日期
    if trip.depart_date < today:
        findings.append(
            PolicyFinding(
                code="DATE_IN_PAST",
                severity="block",
                message=f"出发日期 {trip.depart_date} 早于今天，无法预订。",
                field="depart_date",
                expected="今天或之后的日期",
            )
        )
    if trip.trip_type is TripType.ROUND_TRIP and trip.return_date is None:
        findings.append(
            PolicyFinding(
                code="RETURN_DATE_REQUIRED",
                severity="block",
                message="往返行程缺少返程日期。",
                field="return_date",
                expected="YYYY-MM-DD",
            )
        )
    if trip.return_date and trip.return_date < trip.depart_date:
        findings.append(
            PolicyFinding(
                code="DATE_ORDER_INVALID",
                severity="block",
                message="返程日期早于出发日期。",
                field="return_date",
                expected=f"不早于 {trip.depart_date}",
            )
        )

    # ------------------------------------------------------------------ 提前预订
    lead_days = (trip.depart_date - today).days
    min_days = policy.advance_days_international if trip.international else policy.advance_days_domestic
    if 0 <= lead_days < min_days:
        findings.append(
            PolicyFinding(
                code="ADVANCE_DAYS",
                severity="warn",
                message=(
                    f"距出发仅 {lead_days} 天，低于"
                    f"{'国际' if trip.international else '国内'}行程要求的提前 {min_days} 天。"
                ),
                field="depart_date",
                expected=f"至少提前 {min_days} 天",
            )
        )

    # ------------------------------------------------------------------ 舱位
    allowed = allowed_cabin(policy, trip)
    if CABIN_RANK[trip.cabin_class] > CABIN_RANK[allowed]:
        findings.append(
            PolicyFinding(
                code="CABIN_OVER_STANDARD",
                severity="warn",
                message=(
                    f"{trip.traveler_level.value} 职级"
                    f"{'国际' if trip.international else '国内'}行程可订最高舱位为"
                    f"{CABIN_LABELS[allowed]}，当前为{CABIN_LABELS[trip.cabin_class]}。"
                ),
                field="cabin_class",
                expected=CABIN_LABELS[allowed],
            )
        )

    # ------------------------------------------------------------------ 住宿
    if trip.hotel_needed and hotel_nightly_cny is not None:
        tier = infer_city_tier(trip.destination_city)
        cap = policy.hotel_caps[tier][trip.traveler_level]
        if hotel_nightly_cny > cap:
            nights = 1
            if trip.return_date:
                nights = max(1, (trip.return_date - trip.depart_date).days)
            delta = (hotel_nightly_cny - cap) * nights
            over_standard += delta
            findings.append(
                PolicyFinding(
                    code="HOTEL_OVER_STANDARD",
                    severity="warn",
                    message=(
                        f"{CITY_TIER_LABELS[tier]}住宿标准为 {cap} 元/晚，"
                        f"当前 {hotel_nightly_cny} 元/晚，{nights} 晚超标 {delta} 元。"
                    ),
                    field="hotel_nightly_cny",
                    expected=f"≤ {cap} 元/晚",
                    over_standard_cny=delta,
                )
            )

    # ------------------------------------------------------------------ 预算
    if total_cny is not None and trip.budget_cny is not None and total_cny > trip.budget_cny:
        findings.append(
            PolicyFinding(
                code="BUDGET_EXCEEDED",
                severity="warn",
                message=f"行程预估 {total_cny} 元，超出预算 {trip.budget_cny} 元。",
                field="budget_cny",
                expected=f"≤ {trip.budget_cny} 元",
                over_standard_cny=total_cny - trip.budget_cny,
            )
        )

    # ------------------------------------------------------------------ 硬约束
    if trip.international and not trip.passport_ready:
        findings.append(
            PolicyFinding(
                code="PASSPORT_MISSING",
                severity="block",
                message="国际行程需要有效护照/签证信息，当前画像显示证件未就绪。",
                field="passport_ready",
                expected="先补全证件信息",
            )
        )
    if trip.destination_city in policy.restricted_destinations:
        findings.append(
            PolicyFinding(
                code="RESTRICTED_DESTINATION",
                severity="block",
                message=f"{trip.destination_city} 属于受限目的地，需合规审批后才能预订。",
                field="destination_city",
                expected="更换目的地或走特批流程",
            )
        )
    if trip.international and policy.international_requires_compliance:
        findings.append(
            PolicyFinding(
                code="INTERNATIONAL_COMPLIANCE",
                severity="info",
                message="国际行程需合规审核，请预留审批时间。",
            )
        )

    compliant = not any(f.severity == "block" for f in findings)
    chain = build_approval_chain(
        policy,
        trip,
        total_cny=total_cny or Decimal("0"),
        findings=findings,
        over_standard=over_standard,
    )
    return ComplianceResult(
        compliant=compliant,
        findings=findings,
        required_roles=chain,
        approval_path=chain,
        over_standard_total_cny=over_standard,
        policy_version=policy.version,
    )


def build_approval_chain(
    policy: TravelPolicy,
    trip: TripRequest,
    *,
    total_cny: Decimal,
    findings: list[PolicyFinding],
    over_standard: Decimal,
) -> list[str]:
    """按金额与风险拼审批链。顺序即审批顺序，去重但保留首次出现的位置。"""
    codes = {f.code for f in findings}
    roles: list[str] = ["部门负责人", "差旅管理员"]

    if total_cny >= policy.finance_review_above_cny[trip.traveler_level] or over_standard > 0:
        roles.append("财务BP")
    if trip.international and policy.international_requires_compliance:
        roles.append("合规审核")
    if trip.international and not trip.passport_ready:
        roles.append("签证专员")
    if "CABIN_OVER_STANDARD" in codes or total_cny >= policy.executive_review_above_cny[trip.traveler_level]:
        roles.append("分管高管")
    if over_standard > 0 and policy.over_standard_reason_required:
        roles.append("费用合规复核")

    seen: set[str] = set()
    ordered: list[str] = []
    for role in roles:
        if role not in seen:
            seen.add(role)
            ordered.append(role)
    return ordered


__all__ = [
    "allowed_cabin",
    "build_approval_chain",
    "default_travel_policy",
    "evaluate_trip",
]
