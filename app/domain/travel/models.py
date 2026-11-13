"""差旅领域模型。

与采购域的区别不在字段多少，而在**约束的性质**：
采购的核心是预算、供应商与合同；差旅的核心是"标准 + 时效 + 退改"。
舱位与住宿标准按职级和城市等级确定，报价有有效期，预订后还有退改成本。
这些约束必须是结构化的、可单测的，不能交给模型自由发挥。
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TravelerLevel(str, Enum):
    STAFF = "staff"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"


class CabinClass(str, Enum):
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


CABIN_RANK: dict[CabinClass, int] = {
    CabinClass.ECONOMY: 1,
    CabinClass.PREMIUM_ECONOMY: 2,
    CabinClass.BUSINESS: 3,
    CabinClass.FIRST: 4,
}

CABIN_LABELS: dict[CabinClass, str] = {
    CabinClass.ECONOMY: "经济舱",
    CabinClass.PREMIUM_ECONOMY: "超级经济舱",
    CabinClass.BUSINESS: "公务舱",
    CabinClass.FIRST: "头等舱",
}


class TripType(str, Enum):
    ONE_WAY = "one_way"
    ROUND_TRIP = "round_trip"
    MULTI_CITY = "multi_city"


class CityTier(str, Enum):
    TIER_1 = "tier_1"
    NEW_TIER_1 = "new_tier_1"
    OTHER = "other"


CITY_TIER_LABELS: dict[CityTier, str] = {
    CityTier.TIER_1: "一线城市",
    CityTier.NEW_TIER_1: "新一线城市",
    CityTier.OTHER: "其他城市",
}

_TIER_1_CITIES = frozenset({"北京", "上海", "广州", "深圳"})
_NEW_TIER_1_CITIES = frozenset(
    {"成都", "杭州", "重庆", "武汉", "西安", "苏州", "天津", "南京", "郑州", "长沙", "东莞", "宁波", "佛山", "合肥", "青岛"}
)


def infer_city_tier(city: str) -> CityTier:
    """按城市名推断差标档位。生产环境应改为城市主数据，不在代码里硬编码。"""
    name = (city or "").strip()
    if name in _TIER_1_CITIES:
        return CityTier.TIER_1
    if name in _NEW_TIER_1_CITIES:
        return CityTier.NEW_TIER_1
    return CityTier.OTHER


class TripPurpose(str, Enum):
    CLIENT = "client"
    INTERNAL = "internal"
    TRAINING = "training"
    CONFERENCE = "conference"
    OTHER = "other"


class OfferKind(str, Enum):
    FLIGHT = "flight"
    HOTEL = "hotel"


class BookingStatus(str, Enum):
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    PARTIAL = "partial"


class TripRequest(BaseModel):
    """一次出差的规范化请求。由 L2 草稿 + L3 画像组装，不由模型直接填写主体字段。"""

    trip_id: str
    traveler_id: str
    traveler_level: TravelerLevel
    department: str
    tenant_id: str = "default"
    origin_city: str
    destination_city: str
    depart_date: date
    return_date: date | None = None
    trip_type: TripType = TripType.ONE_WAY
    purpose: TripPurpose = TripPurpose.INTERNAL
    travelers: int = Field(1, ge=1, le=20)
    cabin_class: CabinClass = CabinClass.ECONOMY
    hotel_needed: bool = False
    hotel_star: int = Field(4, ge=1, le=5)
    budget_cny: Decimal | None = None
    cost_center: str | None = None
    project_code: str | None = None
    international: bool = False
    passport_ready: bool = True
    estimated_flight_minutes: int = Field(150, ge=30, le=1200)
    reason: str = ""


class TravelOffer(BaseModel):
    """供应方报价快照。`expires_at` 与价格一起进入确认门，避免"确认时价格已变"。"""

    offer_id: str
    kind: OfferKind
    supplier: str
    title: str
    price_cny: Decimal
    currency: str = "CNY"
    expires_at: datetime
    refundable: bool = False
    cancellation_policy: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class ItinerarySegment(BaseModel):
    kind: OfferKind
    offer_id: str
    title: str
    supplier: str
    price_cny: Decimal
    details: dict[str, Any] = Field(default_factory=dict)


class Itinerary(BaseModel):
    plan_id: str
    trip_id: str
    segments: list[ItinerarySegment]
    total_cny: Decimal
    currency: str = "CNY"
    created_at: datetime
    expires_at: datetime
    notes: str = ""


class PolicyFinding(BaseModel):
    """一条差标结论。`severity=block` 表示不补信息/不换方案就无法继续。"""

    code: str
    severity: str = "warn"
    message: str
    field: str | None = None
    expected: str | None = None
    over_standard_cny: Decimal | None = None


class ComplianceResult(BaseModel):
    compliant: bool
    findings: list[PolicyFinding] = Field(default_factory=list)
    required_roles: list[str] = Field(default_factory=list)
    approval_path: list[str] = Field(default_factory=list)
    over_standard_total_cny: Decimal = Decimal("0")
    policy_version: str = ""

    @property
    def blocking(self) -> list[PolicyFinding]:
        return [f for f in self.findings if f.severity == "block"]

    @property
    def warnings(self) -> list[PolicyFinding]:
        return [f for f in self.findings if f.severity != "block"]


class BookingRecord(BaseModel):
    booking_id: str
    tenant_id: str = ""
    trip_id: str
    offer_id: str
    kind: OfferKind
    status: BookingStatus
    supplier: str
    title: str
    price_cny: Decimal
    traveler_id: str
    order_no: str
    idempotency_key: str = ""
    created_at: datetime
    note: str = ""


class TravelPolicy(BaseModel):
    """公司差旅制度。版本化，改版后旧记忆与旧知识片段不得参与判断。"""

    policy_id: str
    company_id: str
    version: str
    effective_from: date
    #: 职级 -> 国内航线可订的最高舱位。
    domestic_cabin: dict[TravelerLevel, CabinClass]
    #: 职级 -> 国际/长航线可订的最高舱位。
    international_cabin: dict[TravelerLevel, CabinClass]
    #: 飞行时长达到该阈值时，按 `long_haul_cabin` 放宽一档。
    long_haul_minutes: int = 360
    long_haul_cabin: dict[TravelerLevel, CabinClass] = Field(default_factory=dict)
    #: 城市等级 -> 职级 -> 每晚上限（元）。
    hotel_caps: dict[CityTier, dict[TravelerLevel, Decimal]]
    advance_days_domestic: int = 3
    advance_days_international: int = 7
    #: 总费用达到该阈值时需要财务 BP 复核。
    finance_review_above_cny: dict[TravelerLevel, Decimal]
    #: 总费用达到该阈值时需要分管高管审批。
    executive_review_above_cny: dict[TravelerLevel, Decimal]
    restricted_destinations: frozenset[str] = frozenset()
    international_requires_compliance: bool = True
    receipt_required_above_cny: Decimal = Decimal("500")
    #: 超标但仍可提交时，必须由用户填写理由。
    over_standard_reason_required: bool = True
