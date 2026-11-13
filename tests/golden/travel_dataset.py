"""差旅金标评测集。

只放确定性用例：意图快车道、槽位解析、差标合规、审批链。这些不调用模型，
可以直接作为 CI 门禁。需要模型参与的端到端质量评估在 `evals/` 里单独跑。

新增用例请追加，不要为了通过而调低阈值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from app.domain.travel.models import CabinClass, TravelerLevel, TripPurpose, TripRequest, TripType

REFERENCE_TODAY = date(2026, 11, 11)


@dataclass(frozen=True)
class IntentCase:
    text: str
    expected: str


INTENT_CASES: tuple[IntentCase, ...] = (
    IntentCase("帮我查一下明天上海到北京的机票", "flight_search"),
    IntentCase("帮我订一张去深圳的机票", "booking"),
    IntentCase("北京有哪些协议酒店", "hotel_search"),
    IntentCase("查一下酒店", "hotel_search"),
    IntentCase("公司差旅住宿标准是多少", "policy"),
    IntentCase("住宿超标了怎么办", "policy"),
    IntentCase("帮我安排下周去北京出差", "trip_planning"),
    IntentCase("提交审批", "approval"),
    IntentCase("我的订单到哪一步了", "trip_status"),
    IntentCase("改签到明天", "trip_status"),
    IntentCase("差旅管理办法在哪", "rag"),
    IntentCase("签证要求", "rag"),
)

#: 规则快车道的最低准确率。
MIN_INTENT_ACCURACY = 0.85

#: 这些输入**不应该**被规则快车道直接放行：要么没命中，要么置信度低于 0.85。
SLOW_LANE_TEXTS: tuple[str, ...] = (
    "安排一下",
    "看看这个",
    "这个可以吗",
    "帮我弄一下",
)


@dataclass(frozen=True)
class SlotCase:
    name: str
    raw: dict[str, object]
    expected: dict[str, object]


SLOT_CASES: tuple[SlotCase, ...] = (
    SlotCase(
        "iso_date_and_myriad_amount",
        {"depart_date": "2026-12-10", "budget_cny": "1.5万"},
        {"depart_date": "2026-12-10", "budget_cny": "15000.00"},
    ),
    SlotCase(
        "relative_weekday",
        {"depart_date": "下周五"},
        {"depart_date": "2026-11-20"},
    ),
    SlotCase(
        "business_days",
        {"depart_date": "3个工作日后"},
        {"depart_date": "2026-11-16"},
    ),
    SlotCase(
        "amount_with_unit_and_separator",
        {"budget_cny": "18,800元"},
        {"budget_cny": "18800.00"},
    ),
    SlotCase(
        "cabin_and_boolean",
        {"cabin_class": "business", "hotel_needed": "需要"},
        {"cabin_class": "business", "hotel_needed": True},
    ),
    SlotCase(
        "purpose_alias",
        {"purpose": "拜访客户"},
        {"purpose": "client"},
    ),
)

MIN_SLOT_ACCURACY = 1.0


@dataclass(frozen=True)
class PolicyCase:
    name: str
    level: TravelerLevel
    destination: str
    days_ahead: int = 10
    cabin: CabinClass = CabinClass.ECONOMY
    hotel_nightly_cny: Decimal | None = None
    hotel_needed: bool = False
    international: bool = False
    passport_ready: bool = True
    total_cny: Decimal | None = None
    expected_codes: tuple[str, ...] = ()
    expected_roles: tuple[str, ...] = ()
    expect_compliant: bool = True


POLICY_CASES: tuple[PolicyCase, ...] = (
    PolicyCase(
        name="manager_business_cabin_over_standard",
        level=TravelerLevel.MANAGER,
        destination="北京",
        cabin=CabinClass.BUSINESS,
        expected_codes=("CABIN_OVER_STANDARD",),
        expected_roles=("分管高管",),
    ),
    PolicyCase(
        name="manager_beijing_hotel_over_cap",
        level=TravelerLevel.MANAGER,
        destination="北京",
        hotel_needed=True,
        hotel_nightly_cny=Decimal("1200"),
        total_cny=Decimal("3200"),
        expected_codes=("HOTEL_OVER_STANDARD",),
        expected_roles=("财务BP", "费用合规复核"),
    ),
    PolicyCase(
        name="international_without_passport_blocks",
        level=TravelerLevel.STAFF,
        destination="东京",
        international=True,
        passport_ready=False,
        expected_codes=("PASSPORT_MISSING",),
        expected_roles=("合规审核", "签证专员"),
        expect_compliant=False,
    ),
    PolicyCase(
        name="short_lead_time_warns",
        level=TravelerLevel.STAFF,
        destination="北京",
        days_ahead=1,
        expected_codes=("ADVANCE_DAYS",),
    ),
    PolicyCase(
        name="restricted_destination_blocks",
        level=TravelerLevel.DIRECTOR,
        destination="叙利亚",
        international=True,
        expected_codes=("RESTRICTED_DESTINATION",),
        expect_compliant=False,
    ),
    PolicyCase(
        name="executive_high_amount_needs_executive",
        level=TravelerLevel.EXECUTIVE,
        destination="上海",
        total_cny=Decimal("90000"),
        expected_codes=(),
        expected_roles=("财务BP", "分管高管"),
    ),
)

MIN_POLICY_RECALL = 1.0
MIN_APPROVAL_PATH_ACCURACY = 1.0


@dataclass(frozen=True)
class ApprovalCase:
    name: str
    level: TravelerLevel
    total_cny: Decimal
    expected_roles: tuple[str, ...]
    international: bool = False


APPROVAL_CASES: tuple[ApprovalCase, ...] = (
    ApprovalCase("manager_below_finance_threshold", TravelerLevel.MANAGER, Decimal("5000"), ("部门负责人", "差旅管理员")),
    ApprovalCase("manager_at_finance_threshold", TravelerLevel.MANAGER, Decimal("6000"), ("部门负责人", "差旅管理员", "财务BP")),
    ApprovalCase("manager_at_executive_threshold", TravelerLevel.MANAGER, Decimal("20000"), ("部门负责人", "差旅管理员", "财务BP", "分管高管")),
    ApprovalCase("director_at_executive_threshold", TravelerLevel.DIRECTOR, Decimal("40000"), ("部门负责人", "差旅管理员", "财务BP", "分管高管")),
    ApprovalCase("international_adds_compliance", TravelerLevel.MANAGER, Decimal("5000"), ("部门负责人", "差旅管理员", "合规审核"), international=True),
)


def build_trip(case: PolicyCase | ApprovalCase, today: date = REFERENCE_TODAY) -> TripRequest:
    return TripRequest(
        trip_id="trip_case",
        traveler_id="E10086",
        traveler_level=case.level,
        department="市场部",
        origin_city="上海",
        destination_city=getattr(case, "destination", "东京" if case.international else "北京"),
        depart_date=today + timedelta(days=getattr(case, "days_ahead", 10)),
        trip_type=TripType.ONE_WAY,
        purpose=TripPurpose.CLIENT,
        cabin_class=getattr(case, "cabin", CabinClass.ECONOMY),
        hotel_needed=getattr(case, "hotel_needed", False),
        international=case.international,
        passport_ready=getattr(case, "passport_ready", True),
        budget_cny=None,
    )


@dataclass
class MetricReport:
    name: str
    total: int = 0
    passed: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 1.0

    def record(self, ok: bool, detail: str = "") -> None:
        self.total += 1
        if ok:
            self.passed += 1
        elif detail:
            self.failures.append(detail)

    def summary(self) -> str:
        head = f"{self.name}: {self.passed}/{self.total} = {self.score:.2%}"
        if not self.failures:
            return head
        return head + "\n  - " + "\n  - ".join(self.failures[:10])
