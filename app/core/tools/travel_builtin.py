"""差旅工具集：搜索、差标校验、行程规划、预订、退订、审批提交。

与采购工具共享同一套注册表与确认门，差异在于业务语义：

- `search_flights` / `search_hotels` 返回**带有效期的报价快照**，不是最终价格；
- `plan_trip` 产出 `last_plan`，预订与审批都基于这份快照；
- `book_trip` 是组合写操作：任一航段失败会回滚已成功的订单（补偿），
  不允许出现"机票订好了、酒店失败但用户以为都成了"；
- 所有主体字段（工号/职级/部门/租户）由服务端注入，模型看不到也改不了。
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.memory.slots import SlotSource
from app.core.logging import get_logger
from app.core.security.principal import Scope
from app.core.tools.coercion import parse_cn_date, parse_cny, parse_positive_int
from app.core.tools.errors import FieldIssue, ToolError, ToolErrorCode
from app.core.tools.registry import ToolRegistry
from app.core.tools.spec import ToolAccess, ToolContext, ToolReturn, ToolSpec
from app.domain.travel.models import (
    CABIN_LABELS,
    CITY_TIER_LABELS,
    CabinClass,
    TravelOffer,
    TravelerLevel,
    TripType,
    infer_city_tier,
)
from app.domain.travel.policy import allowed_cabin, default_travel_policy, evaluate_trip
from app.infrastructure.suppliers.mock_travel import SupplierError

logger = get_logger(__name__)

IDENTITY_FIELDS: frozenset[str] = frozenset(
    {"traveler_id", "traveler_level", "department", "tenant_id"}
)

#: 报价变化超过该比例时必须中止确认，让用户按新价格重新确认。
PRICE_TOLERANCE = Decimal("0.02")


class _IdentityArgs(BaseModel):
    traveler_id: str = Field("", description="服务端注入：出差人工号")
    traveler_level: TravelerLevel = Field(TravelerLevel.STAFF, description="服务端注入：职级")
    department: str = Field("", description="服务端注入：部门")
    tenant_id: str = Field("default", description="服务端注入：法人/租户")


# --------------------------------------------------------------------------- 搜索


class SearchFlightsArgs(_IdentityArgs):
    origin_city: str = Field(..., min_length=1, max_length=64, description="出发城市，如「上海」")
    destination_city: str = Field(..., min_length=1, max_length=64, description="目的城市，如「北京」")
    depart_date: date = Field(..., description="出发日期 YYYY-MM-DD；也接受「下周五」「3 天后」")
    return_date: date | None = Field(None, description="返程日期；往返行程必填")
    cabin_class: CabinClass = Field(CabinClass.ECONOMY, description="舱位，默认经济舱")
    max_price_cny: Decimal | None = Field(None, description="单程最高价（元）")

    @field_validator("depart_date", "return_date", mode="before")
    @classmethod
    def _coerce_date(cls, v: Any) -> Any:
        if v in (None, "") or isinstance(v, date):
            return v or None
        return parse_cn_date(v, field="date")

    @field_validator("max_price_cny", mode="before")
    @classmethod
    def _coerce_price(cls, v: Any) -> Any:
        return None if v in (None, "") else parse_cny(v, field="max_price_cny")


class SearchHotelsArgs(_IdentityArgs):
    city: str = Field(..., min_length=1, max_length=64, description="入住城市")
    check_in: date = Field(..., description="入住日期 YYYY-MM-DD")
    check_out: date = Field(..., description="离店日期 YYYY-MM-DD，必须晚于入住")
    star: int | None = Field(None, ge=1, le=5, description="酒店星级 1-5，缺省不限")
    max_price_cny: Decimal | None = Field(None, description="每晚最高价（元）")

    @field_validator("check_in", "check_out", mode="before")
    @classmethod
    def _coerce_date(cls, v: Any) -> Any:
        return parse_cn_date(v, field="date") if not isinstance(v, date) else v

    @field_validator("max_price_cny", mode="before")
    @classmethod
    def _coerce_price(cls, v: Any) -> Any:
        return None if v in (None, "") else parse_cny(v, field="max_price_cny")


class GetPolicyArgs(_IdentityArgs):
    city: str | None = Field(None, max_length=64, description="按目的地城市查住宿标准，缺省用行程目的地")
    category: str | None = Field(None, max_length=32, description="可选的费用类别，如 flight/hotel")


class CheckComplianceArgs(_IdentityArgs):
    hotel_nightly_cny: Decimal | None = Field(None, description="酒店每晚价格；缺省取已选酒店")

    @field_validator("hotel_nightly_cny", mode="before")
    @classmethod
    def _coerce_price(cls, v: Any) -> Any:
        return None if v in (None, "") else parse_cny(v, field="hotel_nightly_cny")


class PlanTripArgs(_IdentityArgs):
    #: 允许模型在规划时把本轮抽到的槽位一并带上来。工具会先写入 L2 再规划，
    #: 避免"模型说了日期、但服务端草稿里没有"的两份真相。
    origin_city: str | None = Field(None, max_length=64, description="出发城市")
    destination_city: str | None = Field(None, max_length=64, description="目的城市")
    depart_date: date | None = Field(None, description="出发日期 YYYY-MM-DD")
    return_date: date | None = Field(None, description="返程日期 YYYY-MM-DD")
    purpose: str | None = Field(None, max_length=64, description="出差事由")
    travelers: int | None = Field(None, ge=1, le=20, description="出行人数")
    cabin_class: CabinClass | None = Field(None, description="舱位")
    hotel_needed: bool | None = Field(None, description="是否需要住宿")
    hotel_star: int | None = Field(None, ge=1, le=5, description="酒店星级")
    budget_cny: Decimal | None = Field(None, description="预算（元）")
    international: bool | None = Field(None, description="是否国际行程")
    passport_ready: bool | None = Field(None, description="护照/签证是否就绪")
    estimated_flight_minutes: int | None = Field(None, ge=30, le=1200, description="预计飞行分钟数")
    reason: str | None = Field(None, max_length=500, description="补充说明")
    flight_offer_id: str | None = Field(None, description="指定航班报价 ID；缺省按差标选最便宜的合规班次")
    hotel_offer_id: str | None = Field(None, description="指定酒店报价 ID；缺省按差标选")
    notes: str = Field("", max_length=500, description="给方案补充说明")

    @field_validator("depart_date", "return_date", mode="before")
    @classmethod
    def _coerce_date(cls, v: Any) -> Any:
        if v in (None, "") or isinstance(v, date):
            return v or None
        return parse_cn_date(v, field="date")

    @field_validator("budget_cny", mode="before")
    @classmethod
    def _coerce_budget(cls, v: Any) -> Any:
        return None if v in (None, "") else parse_cny(v, field="budget_cny")


class BuildRequestArgs(_IdentityArgs):
    notes: str = Field("", max_length=500, description="出差申请单备注")


class BookTripArgs(_IdentityArgs):
    plan_id: str | None = Field(None, description="要预订的方案 ID；缺省用当前会话最新方案")
    note: str = Field("", max_length=200, description="预订备注")


class CancelBookingArgs(_IdentityArgs):
    booking_id: str = Field(..., min_length=1, description="要取消的订单 ID")
    reason: str = Field(..., min_length=1, max_length=200, description="取消原因")


class SubmitApprovalArgs(_IdentityArgs):
    note: str = Field("", max_length=500, description="审批说明")


class GetBookingStatusArgs(_IdentityArgs):
    booking_id: str = Field(..., min_length=1, description="订单 ID")


class KnowledgeSearchArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="检索关键词，如「住宿标准」「签证要求」")
    top_k: int = Field(5, ge=1, le=10, description="返回片段数")

    @field_validator("top_k", mode="before")
    @classmethod
    def _coerce_top_k(cls, v: Any) -> Any:
        return parse_positive_int(v, field="top_k", maximum=10)


# --------------------------------------------------------------------------- 工具函数


def _supplier(ctx: ToolContext) -> Any:
    if ctx.supplier is None:
        raise ToolError(
            ToolErrorCode.UPSTREAM_ERROR,
            "差旅供应方未接入，当前无法查询实时报价。",
            hint="告知用户供应方接口暂不可用，可先解答差标或审批问题。",
        )
    return ctx.supplier


def _trip(ctx: ToolContext):
    missing = ctx.draft.missing_required()
    if missing:
        raise ToolError(
            ToolErrorCode.MISSING_ARGUMENTS,
            "行程信息不完整，无法继续。",
            hint=ctx.draft.missing_prompt(),
            details={"missing_fields": list(missing)},
        )
    try:
        return ctx.draft.to_trip_request(ctx.profile)
    except (ValueError, TypeError) as exc:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"行程字段格式有误：{exc}",
            hint="检查日期是否为 YYYY-MM-DD，或改用「下周五」这类可解析表述。",
        ) from exc


def _money(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01'))}"


def _offer_line(index: int, offer: TravelOffer) -> str:
    return (
        f"{index}. [{offer.supplier}] {offer.title}｜"
        f"{_money(offer.price_cny)} 元｜"
        f"{'可退' if offer.refundable else '不可退'}｜报价有效期至 {offer.expires_at:%H:%M}"
    )


# --------------------------------------------------------------------------- 只读工具


async def _search_flights(args: SearchFlightsArgs, ctx: ToolContext) -> ToolReturn:
    supplier = _supplier(ctx)
    offers = await supplier.search_flights(
        origin_city=args.origin_city,
        destination_city=args.destination_city,
        depart_date=args.depart_date,
        return_date=args.return_date,
        cabin_class=args.cabin_class,
        max_price_cny=args.max_price_cny,
    )
    if not offers:
        raise ToolError(
            ToolErrorCode.NOT_FOUND,
            f"{args.origin_city}→{args.destination_city} {args.depart_date} 没有查到可订航班。",
            hint="建议用户更换日期或邻近机场；不要编造班次与价格。",
        )
    lines = [f"{args.origin_city}→{args.destination_city} 共 {len(offers)} 个报价（含税）："]
    lines.extend(_offer_line(i, o) for i, o in enumerate(offers, start=1))
    lines.append("报价有有效期，预订前会再次核价。")
    updates: dict[str, Any] = {
        "origin_city": args.origin_city,
        "destination_city": args.destination_city,
        "depart_date": args.depart_date,
    }
    if args.return_date:
        updates["return_date"] = args.return_date
    if args.cabin_class:
        updates["cabin_class"] = args.cabin_class
    return ToolReturn(
        data={"offers": [o.model_dump(mode="json") for o in offers]},
        rendered="\n".join(lines),
        draft_updates=updates,
    )


async def _search_hotels(args: SearchHotelsArgs, ctx: ToolContext) -> ToolReturn:
    if args.check_out <= args.check_in:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENTS,
            "离店日期必须晚于入住日期。",
            hint="确认入住/离店日期后重试。",
            field_issues=[
                # 字段级信息让模型能自己改，而不是原样重试
                FieldIssue(
                    field="check_out",
                    problem="必须晚于入住日期",
                    expected=f"晚于 {args.check_in}",
                )
            ],
        )
    supplier = _supplier(ctx)
    offers = await supplier.search_hotels(
        city=args.city,
        check_in=args.check_in,
        check_out=args.check_out,
        star=args.star,
        max_price_cny=args.max_price_cny,
    )
    if not offers:
        raise ToolError(
            ToolErrorCode.NOT_FOUND,
            f"{args.city} 在所选日期没有符合条件的酒店。",
            hint="放宽星级或价格上限后重试；不要编造酒店。",
        )
    lines = [f"{args.city} {args.check_in} 至 {args.check_out} 共 {len(offers)} 个住宿报价："]
    for i, offer in enumerate(offers, start=1):
        nightly = offer.details.get("nightly_cny", "")
        lines.append(f"{_offer_line(i, offer)}（{nightly} 元/晚）")
    return ToolReturn(
        data={"offers": [o.model_dump(mode="json") for o in offers]},
        rendered="\n".join(lines),
        draft_updates={
            "destination_city": args.city,
            "hotel_needed": True,
            **({"hotel_star": args.star} if args.star else {}),
        },
    )


async def _get_policy(args: GetPolicyArgs, ctx: ToolContext) -> ToolReturn:
    policy = ctx.policy or default_travel_policy()
    level = TravelerLevel(ctx.profile.requester_level.value)
    city = args.city or ctx.draft.value("destination_city") or ""
    tier = infer_city_tier(str(city)) if city else None
    cabin_domestic = policy.domestic_cabin[level]
    cabin_intl = policy.international_cabin[level]

    data: dict[str, Any] = {
        "policy_version": policy.version,
        "level": level.value,
        "city": city,
        "city_tier": tier.value if tier else None,
        "cabin_domestic": cabin_domestic.value,
        "cabin_international": cabin_intl.value,
        "advance_days_domestic": policy.advance_days_domestic,
        "advance_days_international": policy.advance_days_international,
        "finance_review_above_cny": str(policy.finance_review_above_cny[level]),
        "executive_review_above_cny": str(policy.executive_review_above_cny[level]),
    }
    lines = [
        f"差旅标准（制度版本 {policy.version}）",
        f"职级：{level.value}；城市：{city or '未指定'}"
        + (f"（{CITY_TIER_LABELS[tier]}）" if tier else ""),
        f"国内舱位上限：{CABIN_LABELS[cabin_domestic]}；国际/长航线：{CABIN_LABELS[cabin_intl]}",
        f"提前预订：国内 {policy.advance_days_domestic} 天 / 国际 {policy.advance_days_international} 天",
        f"财务复核阈值：{policy.finance_review_above_cny[level]} 元；"
        f"高管审批阈值：{policy.executive_review_above_cny[level]} 元",
    ]
    if tier:
        cap = policy.hotel_caps[tier][level]
        data["hotel_cap_cny"] = str(cap)
        lines.append(f"住宿上限：{cap} 元/晚（{CITY_TIER_LABELS[tier]}）")
    return ToolReturn(data=data, rendered="\n".join(lines))


async def _check_compliance(args: CheckComplianceArgs, ctx: ToolContext) -> ToolReturn:
    trip = _trip(ctx)
    policy = ctx.policy or default_travel_policy()
    hotel_nightly: Decimal | None = None
    if args.hotel_nightly_cny is not None:
        hotel_nightly = args.hotel_nightly_cny
    elif ctx.draft.selected_hotel:
        raw = ctx.draft.selected_hotel.get("details", {}).get("nightly_cny")
        hotel_nightly = Decimal(str(raw)) if raw is not None else None

    total_raw = (ctx.draft.last_plan or {}).get("total_cny")
    total = Decimal(str(total_raw)) if total_raw is not None else None
    result = evaluate_trip(
        policy,
        trip,
        hotel_nightly_cny=hotel_nightly,
        total_cny=total,
        today=ctx.today,
    )
    lines = [f"差标校验（制度版本 {result.policy_version}）"]
    lines.append("结论：符合差标" if result.compliant else "结论：存在阻塞项，需先处理")
    if result.findings:
        for finding in result.findings:
            flag = {"block": "✗", "warn": "!", "info": "·"}.get(finding.severity, "·")
            lines.append(f"{flag} {finding.message}")
    if result.over_standard_total_cny:
        lines.append(f"超标金额合计：{_money(result.over_standard_total_cny)} 元")
    lines.append("审批链：" + " → ".join(result.required_roles))
    return ToolReturn(
        data={"compliance": result.model_dump(mode="json")},
        rendered="\n".join(lines),
        draft_facts={},  # 不把差标结论写进槽位，避免被摘要压缩成"事实"
    )


async def _search_knowledge(args: KnowledgeSearchArgs, ctx: ToolContext) -> ToolReturn:
    if ctx.knowledge is None:
        raise ToolError(
            ToolErrorCode.UPSTREAM_ERROR,
            "知识库未接入，无法检索制度原文。",
            hint="改为给出通用差标说明，并提示以公司制度为准。",
        )
    chunks = await ctx.knowledge.search_active(
        args.query, top_k=args.top_k, tenant_id=ctx.tenant_id
    )
    if not chunks:
        raise ToolError(
            ToolErrorCode.NOT_FOUND,
            "知识库中没有找到相关制度片段。",
            hint="换关键词重试，或提示用户查阅差旅管理办法原文。",
        )
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        title = chunk.get("title") or chunk.get("doc_id") or "未命名文档"
        content = str(chunk.get("content") or "")[:900]
        blocks.append(f"[{index}] {title}\n{content}")
    rendered = "<knowledge>\n" + "\n\n".join(blocks) + "\n</knowledge>"
    return ToolReturn(
        data={"chunks": chunks},
        rendered=rendered,
    )


# --------------------------------------------------------------------------- 规划


async def _plan_trip(args: PlanTripArgs, ctx: ToolContext) -> ToolReturn:
    slot_fields = (
        "origin_city",
        "destination_city",
        "depart_date",
        "return_date",
        "purpose",
        "travelers",
        "cabin_class",
        "hotel_needed",
        "hotel_star",
        "budget_cny",
        "international",
        "passport_ready",
        "estimated_flight_minutes",
        "reason",
    )
    updates = {
        field: getattr(args, field)
        for field in slot_fields
        if getattr(args, field, None) is not None
    }
    if updates:
        # 先落 L2，再基于同一份状态规划：避免"模型说了一个日期、草稿里是另一个"。
        ctx.draft.merge(updates, source=SlotSource.USER, turn=ctx.turn)

    trip = _trip(ctx)
    policy = ctx.policy or default_travel_policy()
    supplier = _supplier(ctx)

    allowed = allowed_cabin(policy, trip)
    flights = await supplier.search_flights(
        origin_city=trip.origin_city,
        destination_city=trip.destination_city,
        depart_date=trip.depart_date,
        return_date=trip.return_date,
        cabin_class=trip.cabin_class,
    )
    outbound = [f for f in flights if f.details.get("direction") == "outbound"]
    inbound = [f for f in flights if f.details.get("direction") == "return"]
    if not outbound:
        raise ToolError(ToolErrorCode.NOT_FOUND, "没有查到符合条件的去程航班。")

    chosen_out = _pick_offer(outbound, args.flight_offer_id)
    segments: list[TravelOffer] = [chosen_out]
    if trip.trip_type is TripType.ROUND_TRIP:
        if not inbound:
            raise ToolError(ToolErrorCode.NOT_FOUND, "没有查到符合条件的返程航班。")
        segments.append(_pick_offer(inbound, None))

    hotel: TravelOffer | None = None
    hotel_nightly: Decimal | None = None
    if trip.hotel_needed:
        check_out = trip.return_date or (trip.depart_date + timedelta(days=1))
        hotels = await supplier.search_hotels(
            city=trip.destination_city,
            check_in=trip.depart_date,
            check_out=check_out,
            star=trip.hotel_star,
        )
        if not hotels:
            raise ToolError(ToolErrorCode.NOT_FOUND, "没有查到符合条件的酒店。")
        tier = infer_city_tier(trip.destination_city)
        cap = policy.hotel_caps[tier][trip.traveler_level]
        within = [h for h in hotels if Decimal(str(h.details.get("nightly_cny", "0"))) <= cap]
        hotel = _pick_offer(within or hotels, args.hotel_offer_id)
        hotel_nightly = Decimal(str(hotel.details.get("nightly_cny", "0")))
        segments.append(hotel)

    total = sum((o.price_cny for o in segments), Decimal("0"))
    compliance = evaluate_trip(
        policy,
        trip,
        hotel_nightly_cny=hotel_nightly,
        total_cny=total,
        today=ctx.today,
    )
    now = datetime.now(timezone.utc)
    expires_at = min((o.expires_at for o in segments), default=now)
    plan_id = "plan_" + hashlib.sha256(
        f"{ctx.draft.fingerprint()}|{','.join(o.offer_id for o in segments)}".encode("utf-8")
    ).hexdigest()[:16]

    plan = {
        "plan_id": plan_id,
        "trip_id": trip.trip_id,
        "status": "planned",
        "segments": [o.model_dump(mode="json") for o in segments],
        "total_cny": _money(total),
        "currency": "CNY",
        "created_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "approval_path": compliance.approval_path,
        "compliance": compliance.model_dump(mode="json"),
        "notes": args.notes,
        "cabin_allowed": allowed.value,
    }

    lines = [
        f"行程方案 {plan_id}（报价有效期至 {expires_at:%Y-%m-%d %H:%M} UTC）",
    ]
    for i, segment in enumerate(segments, start=1):
        lines.append(f"{i}. [{segment.supplier}] {segment.title}｜{_money(segment.price_cny)} 元")
    lines.append(f"合计：{_money(total)} 元")
    if compliance.findings:
        lines.append("差标提示：")
        for finding in compliance.findings:
            lines.append(f"- {finding.message}")
    lines.append("审批链：" + " → ".join(compliance.approval_path))
    lines.append("确认方案后可发起预订；预订前会再次核价。")

    return ToolReturn(
        data={"plan": plan},
        rendered="\n".join(lines),
        draft_attributes={
            "last_plan": plan,
            "selected_flight": chosen_out.model_dump(mode="json"),
            "selected_hotel": hotel.model_dump(mode="json") if hotel else None,
        },
    )


def _pick_offer(offers: list[TravelOffer], offer_id: str | None) -> TravelOffer:
    if offer_id:
        for offer in offers:
            if offer.offer_id == offer_id:
                return offer
        raise ToolError(
            ToolErrorCode.NOT_FOUND,
            f"指定的报价 {offer_id} 不在当前搜索结果中。",
            hint="重新搜索后从返回的 offer_id 中选择，不要复用上一轮的报价。",
        )
    return min(offers, key=lambda o: o.price_cny)


async def _build_request(args: BuildRequestArgs, ctx: ToolContext) -> ToolReturn:
    trip = _trip(ctx)
    plan = ctx.draft.last_plan or {}
    total = plan.get("total_cny")
    approval_path = plan.get("approval_path") or []
    if not approval_path:
        compliance = evaluate_trip(
            ctx.policy or default_travel_policy(),
            trip,
            total_cny=Decimal(str(total)) if total else None,
            today=ctx.today,
        )
        approval_path = compliance.approval_path

    request_no = "TR" + hashlib.sha256(ctx.draft.fingerprint().encode()).hexdigest()[:10].upper()
    request = {
        "request_no": request_no,
        "traveler_id": trip.traveler_id,
        "traveler_level": trip.traveler_level.value,
        "department": trip.department,
        "origin_city": trip.origin_city,
        "destination_city": trip.destination_city,
        "depart_date": trip.depart_date.isoformat(),
        "return_date": trip.return_date.isoformat() if trip.return_date else None,
        "purpose": trip.purpose.value,
        "estimated_total_cny": total,
        "cost_center": trip.cost_center,
        "approval_path": approval_path,
        "notes": args.notes,
    }
    lines = [
        f"出差申请单 {request_no}（草稿，未提交）",
        f"出差人：{trip.traveler_id}（{trip.traveler_level.value}）｜部门：{trip.department}",
        f"行程：{trip.origin_city} → {trip.destination_city}"
        f"｜{trip.depart_date}"
        + (f" 至 {trip.return_date}" if trip.return_date else ""),
        f"事由：{trip.purpose.value}",
    ]
    if total:
        lines.append(f"预估费用：{total} 元")
    if trip.cost_center:
        lines.append(f"成本中心：{trip.cost_center}")
    lines.append("审批链：" + " → ".join(approval_path))
    return ToolReturn(
        data={"request": request},
        rendered="\n".join(lines),
        draft_attributes={"last_request": request},
    )


# --------------------------------------------------------------------------- 写操作


async def _persist_booking(ctx: ToolContext, record: Any) -> None:
    """订单落库失败不回滚已出票的订单：供应方侧事实优先，落库失败告警后可对账补偿。"""
    if ctx.booking_repo is None:
        return
    try:
        await ctx.booking_repo.save(record)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "booking.persist_failed",
            booking_id=getattr(record, "booking_id", ""),
            error=str(exc),
        )


async def _book_trip(args: BookTripArgs, ctx: ToolContext) -> ToolReturn:
    supplier = _supplier(ctx)
    plan = ctx.draft.last_plan
    if not plan:
        raise ToolError(
            ToolErrorCode.PRECONDITION_NOT_MET,
            "当前会话没有可预订的行程方案。",
            hint="先调用 plan_trip 生成方案，再预订。",
        )
    if args.plan_id and args.plan_id != plan.get("plan_id"):
        raise ToolError(
            ToolErrorCode.PRECONDITION_NOT_MET,
            f"方案 {args.plan_id} 不是当前会话的最新方案。",
            hint=f"当前方案是 {plan.get('plan_id')}，请让用户重新确认。",
        )
    if datetime.fromisoformat(plan["expires_at"]) <= datetime.now(timezone.utc):
        raise ToolError(
            ToolErrorCode.PRECONDITION_NOT_MET,
            "方案中的报价已过期。",
            hint="重新调用 plan_trip 获取新报价后再预订。",
        )
    blocking = [f for f in plan.get("compliance", {}).get("findings", []) if f.get("severity") == "block"]
    if blocking:
        raise ToolError(
            ToolErrorCode.PRECONDITION_NOT_MET,
            "方案存在阻塞项，不能直接预订：" + "；".join(f["message"] for f in blocking),
            hint="先补齐证件信息或更换目的地。",
        )

    booked = []
    try:
        for segment in plan["segments"]:
            record = await supplier.book(
                segment["offer_id"],
                trip_id=plan["trip_id"],
                traveler_id=ctx.profile.subject_id,
                tenant_id=ctx.tenant_id,
                idempotency_key=(
                    f"{ctx.tenant_id}:{ctx.profile.subject_id}:{plan['plan_id']}:{segment['offer_id']}"
                ),
            )
            booked.append(record)
            await _persist_booking(ctx, record)
    except SupplierError as exc:
        compensated = []
        for record in booked:
            try:
                await supplier.cancel(record.booking_id, reason="组合预订失败自动回滚")
                compensated.append(record.booking_id)
            except SupplierError:
                pass
        raise ToolError(
            ToolErrorCode.UPSTREAM_ERROR,
            f"预订失败：{exc.message}"
            + (f"；已回滚 {len(compensated)} 个已成功订单。" if compensated else ""),
            hint="告知用户未完成预订，不要声称已出票；可换班次重试。",
            details={
                "failed_offer_id": segment["offer_id"],
                "booked": [r.booking_id for r in booked],
                "compensated": compensated,
            },
        ) from exc

    booking_ids = [r.booking_id for r in booked]
    updated_plan = dict(plan)
    updated_plan["status"] = "booked"
    updated_plan["bookings"] = [r.model_dump(mode="json") for r in booked]
    lines = ["预订成功："]
    for record in booked:
        lines.append(
            f"- {record.title}｜订单号 {record.order_no}｜{_money(record.price_cny)} 元｜{record.supplier}"
        )
    lines.append("行程单与发票可在订单详情中获取。")
    return ToolReturn(
        data={"bookings": [r.model_dump(mode="json") for r in booked], "booking_ids": booking_ids},
        rendered="\n".join(lines),
        draft_attributes={"booking_ids": booking_ids, "last_plan": updated_plan},
    )


async def _cancel_booking(args: CancelBookingArgs, ctx: ToolContext) -> ToolReturn:
    supplier = _supplier(ctx)
    try:
        record = await supplier.cancel(args.booking_id, reason=args.reason)
    except SupplierError as exc:
        raise ToolError(
            ToolErrorCode.NOT_FOUND,
            exc.message,
            hint="确认订单号是否正确；不要重复取消同一订单。",
        ) from exc
    await _persist_booking(ctx, record)
    remaining = [b for b in ctx.draft.booking_ids if b != args.booking_id]
    lines = [
        f"已取消：{record.title}（订单 {record.order_no}）",
        f"状态：{record.status.value}",
    ]
    if record.note:
        lines.append(record.note)
    return ToolReturn(
        data={"booking": record.model_dump(mode="json")},
        rendered="\n".join(lines),
        draft_attributes={"booking_ids": remaining},
    )


async def _submit_approval(args: SubmitApprovalArgs, ctx: ToolContext) -> ToolReturn:
    trip = _trip(ctx)
    plan = ctx.draft.last_plan
    if not plan:
        raise ToolError(
            ToolErrorCode.PRECONDITION_NOT_MET,
            "还没有行程方案，无法提交审批。",
            hint="先调用 plan_trip 生成方案。",
        )
    if ctx.draft.submitted_approval_no:
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            f"该行程已提交审批，单号 {ctx.draft.submitted_approval_no}。",
            hint="告知用户已提交过，不要重复提交。",
        )
    approval_no = "TRV-" + hashlib.sha256(
        f"{ctx.tenant_id}:{ctx.profile.subject_id}:{plan['plan_id']}".encode()
    ).hexdigest()[:10].upper()
    approval = {
        "approval_no": approval_no,
        "trip_id": trip.trip_id,
        "plan_id": plan["plan_id"],
        "status": "pending",
        "current_stage": (plan.get("approval_path") or ["部门负责人"])[0],
        "approval_path": plan.get("approval_path") or [],
        "note": args.note,
    }
    rendered = (
        f"审批已提交。审批单号 {approval_no}，"
        f"当前节点：{approval['current_stage']}，状态：pending。"
    )
    return ToolReturn(
        data={"approval": approval},
        rendered=rendered,
        draft_attributes={"submitted_approval_no": approval_no},
    )


async def _get_booking_status(args: GetBookingStatusArgs, ctx: ToolContext) -> ToolReturn:
    supplier = _supplier(ctx)
    record = await supplier.get_booking(args.booking_id)
    if record is None:
        raise ToolError(
            ToolErrorCode.NOT_FOUND,
            f"订单 {args.booking_id} 不存在。",
            hint="确认订单号；不要编造订单状态。",
        )
    return ToolReturn(
        data={"booking": record.model_dump(mode="json")},
        rendered=(
            f"订单 {record.order_no}：{record.title}\n"
            f"状态：{record.status.value}｜金额：{_money(record.price_cny)} 元"
            + (f"\n备注：{record.note}" if record.note else "")
        ),
    )


# --------------------------------------------------------------------------- 确认门预览


def _preview_book_trip(args: BookTripArgs, ctx: ToolContext) -> str:
    plan = ctx.draft.last_plan or {}
    lines = [f"即将预订以下行程（方案 {plan.get('plan_id')}）："]
    for segment in plan.get("segments", []):
        lines.append(
            f"- [{segment.get('supplier')}] {segment.get('title')}：{segment.get('price_cny')} 元"
        )
    lines.append(f"合计：{plan.get('total_cny')} 元")
    lines.append("")
    lines.append("报价有有效期，确认时会再次核价；价格变动超过 2% 将中止并请你重新确认。")
    lines.append("预订后按各报价的退改规则执行，部分特价票不可退。")
    if args.note:
        lines.append(f"备注：{args.note}")
    lines.append("确认后才会真正出票/占房。")
    return "\n".join(lines)


def _preview_cancel(args: CancelBookingArgs, ctx: ToolContext) -> str:
    return (
        f"即将取消订单 {args.booking_id}。\n"
        f"取消原因：{args.reason}\n"
        "取消后是否产生退票费取决于该报价的退改规则，确认后不可撤销。"
    )


def _preview_submit(args: SubmitApprovalArgs, ctx: ToolContext) -> str:
    plan = ctx.draft.last_plan or {}
    lines = [f"即将提交出差审批（方案 {plan.get('plan_id')}）："]
    lines.append(f"预估费用：{plan.get('total_cny')} 元")
    lines.append("审批链：" + " → ".join(plan.get("approval_path") or []))
    if args.note:
        lines.append(f"说明：{args.note}")
    lines.append("提交后进入审批流程，撤回需要联系审批人。")
    return "\n".join(lines)


# --------------------------------------------------------------------------- 注册


def register_travel_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="search_flights",
            description="按城市、日期、舱位检索航班报价，返回带有效期的 offer_id 与价格。只读。",
            args_model=SearchFlightsArgs,
            handler=_search_flights,
            access=ToolAccess.READ,
            injected_fields=IDENTITY_FIELDS,
        )
    )
    registry.register(
        ToolSpec(
            name="search_hotels",
            description="按城市与入住日期检索酒店报价，返回每晚价格与退改规则。只读。",
            args_model=SearchHotelsArgs,
            handler=_search_hotels,
            access=ToolAccess.READ,
            injected_fields=IDENTITY_FIELDS,
        )
    )
    registry.register(
        ToolSpec(
            name="get_travel_policy",
            description="查询当前职级的差旅标准：舱位上限、住宿上限、提前预订天数、审批阈值。",
            args_model=GetPolicyArgs,
            handler=_get_policy,
            access=ToolAccess.READ,
            injected_fields=IDENTITY_FIELDS,
        )
    )
    registry.register(
        ToolSpec(
            name="check_travel_compliance",
            description="校验当前行程是否符合差标，返回超标项、超标金额与审批链。",
            args_model=CheckComplianceArgs,
            handler=_check_compliance,
            access=ToolAccess.READ,
            injected_fields=IDENTITY_FIELDS,
            draft_backfill=True,
        )
    )
    registry.register(
        ToolSpec(
            name="search_travel_knowledge",
            description="检索差旅制度、报销标准、签证要求等知识库原文，返回内容仅作参考。",
            args_model=KnowledgeSearchArgs,
            handler=_search_knowledge,
            access=ToolAccess.READ,
            required_scopes=(Scope.KNOWLEDGE_READ,),
        )
    )
    registry.register(
        ToolSpec(
            name="plan_trip",
            description=(
                "根据当前草稿生成差旅方案：选定航班与酒店、计算总价、校验差标并给出审批链。"
                "只生成方案，不预订。"
            ),
            args_model=PlanTripArgs,
            handler=_plan_trip,
            access=ToolAccess.DRAFT,
            injected_fields=IDENTITY_FIELDS,
            required_scopes=(Scope.TRIP_DRAFT,),
            draft_backfill=True,
        )
    )
    registry.register(
        ToolSpec(
            name="build_travel_request",
            description="生成出差申请单草稿，包含行程、预估费用与审批链，不提交审批。",
            args_model=BuildRequestArgs,
            handler=_build_request,
            access=ToolAccess.DRAFT,
            injected_fields=IDENTITY_FIELDS,
            required_scopes=(Scope.TRIP_DRAFT,),
            draft_backfill=True,
        )
    )
    registry.register(
        ToolSpec(
            name="book_trip",
            description=(
                "按当前方案预订机票与酒店。这是不可逆写操作：调用后系统会先展示摘要并等待"
                "用户显式确认，你不能假定已出票。"
            ),
            args_model=BookTripArgs,
            handler=_book_trip,
            access=ToolAccess.WRITE,
            injected_fields=IDENTITY_FIELDS,
            required_scopes=(Scope.TRIP_BOOK,),
            preview=_preview_book_trip,
        )
    )
    registry.register(
        ToolSpec(
            name="cancel_booking",
            description="取消已预订的机票或酒店。不可逆写操作，需用户确认。",
            args_model=CancelBookingArgs,
            handler=_cancel_booking,
            access=ToolAccess.WRITE,
            injected_fields=IDENTITY_FIELDS,
            required_scopes=(Scope.TRIP_BOOK,),
            preview=_preview_cancel,
        )
    )
    registry.register(
        ToolSpec(
            name="submit_travel_approval",
            description="把当前行程方案提交到审批系统。不可逆写操作，需用户确认。",
            args_model=SubmitApprovalArgs,
            handler=_submit_approval,
            access=ToolAccess.WRITE,
            injected_fields=IDENTITY_FIELDS,
            required_scopes=(Scope.TRIP_APPROVE,),
            preview=_preview_submit,
        )
    )
    registry.register(
        ToolSpec(
            name="get_booking_status",
            description="查询订单状态、金额与退改备注。只读。",
            args_model=GetBookingStatusArgs,
            handler=_get_booking_status,
            access=ToolAccess.READ,
            injected_fields=IDENTITY_FIELDS,
        )
    )


def build_travel_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_travel_tools(registry)
    return registry


__all__ = ["PRICE_TOLERANCE", "build_travel_registry", "register_travel_tools"]
