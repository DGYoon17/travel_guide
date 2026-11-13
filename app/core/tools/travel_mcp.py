"""MCP 实时数据工具：车次、自驾、酒店、天气、黄历、航班，以及城际交通对比。

设计要点与 `travel_builtin` 一致：注册即声明（pydantic schema 派生 function-calling），
执行结果同时产出"喂给模型的文本"与"进审计的结构化数据"。差异在于数据来源是外部
MCP 服务，因此：

- **所有工具都是 READ**：查车次/查天气不改变任何业务状态，不该进确认门；
- **失败必须是可读的降级**：`MCPError` 统一翻译成 `ToolError(UPSTREAM_ERROR)`，
  并给出"告知用户实时数据暂不可用，不要编造"的 hint，避免模型用常识补齐车次时刻；
- **参数可从 L2 草稿回填**：模型忘了带 origin/destination 时用草稿里已确认的值，
  而不是反问用户"你要从哪出发"。

`plan_intercity_transport` 是复合工具，对应参考项目里"12306 查询自动附带自驾路线"的
行为：一次调用拿到火车 + 自驾 + （长途时）航班，并给出统一建议，减少 ReAct 轮次。
"""

from __future__ import annotations

import asyncio
from datetime import date as Date
from datetime import timedelta
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.core.logging import get_logger
from app.core.tools.coercion import parse_cn_date, parse_cny
from app.core.tools.errors import FieldIssue, ToolError, ToolErrorCode
from app.core.tools.registry import ToolRegistry
from app.core.tools.spec import ToolAccess, ToolContext, ToolReturn, ToolSpec
from app.domain.travel.geo import estimate_distance
from app.infrastructure.mcp.hub import (
    FLIGHT_SERVER,
    MCPError,
)
from app.infrastructure.mcp.travel_data import (
    Almanac,
    AirportCodeError,
    DrivingRoute,
    FlightSearch,
    HotelSearch,
    LiveTravelData,
    TrainTickets,
    WeatherReport,
)

logger = get_logger(__name__)

#: 单条渲染文本的上限。观测值会进入短期记忆，必须给上下文留预算。
TRAIN_TEXT_LIMIT = 4_500
COMPOSITE_TEXT_LIMIT = 3_200
RAW_TEXT_LIMIT = 2_000

#: 自驾舒适度分档（公里）：超过 300 提示疲劳，超过 500 建议改用高铁/飞机。
DRIVING_COMFORT_KM = 300.0
DRIVING_RECOMMEND_OTHER_KM = 500.0


# --------------------------------------------------------------------------- 参数模型


def _coerce_optional_date(value: Any) -> Any:
    # 注意：参数名 `date` 与 `datetime.date` 同名，所以类型统一用 `Date` 别名，
    # 否则 pydantic 在类体内解析注解时会把字段值（FieldInfo）当成类型。
    if value in (None, "") or isinstance(value, Date):
        return value or None
    return parse_cn_date(value, field="date")


class TrainTicketsArgs(BaseModel):
    origin: str | None = Field(None, max_length=32, description="出发城市，如「上海」；缺省取行程出发地")
    destination: str | None = Field(None, max_length=32, description="到达城市，如「苏州」；缺省取行程目的地")
    date: Date | None = Field(None, description="乘车日期 YYYY-MM-DD；缺省取行程出发日期")

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, value: Any) -> Any:
        return _coerce_optional_date(value)


class DrivingRouteArgs(BaseModel):
    origin: str | None = Field(None, max_length=32, description="出发城市；缺省取行程出发地")
    destination: str | None = Field(None, max_length=32, description="到达城市；缺省取行程目的地")


class LocalHotelSearchArgs(BaseModel):
    city: str | None = Field(None, max_length=32, description="城市；缺省取行程目的地")
    budget_per_night_cny: Decimal | None = Field(
        None, description="每晚预算（元）；缺省按总预算 ÷ 住宿晚数推算"
    )
    star: int | None = Field(None, ge=1, le=5, description="偏好星级 1-5，仅用于生成检索关键词")
    keywords: str | None = Field(None, max_length=64, description="自定义检索关键词，如「杭州 亲子酒店」")

    @field_validator("budget_per_night_cny", mode="before")
    @classmethod
    def _price(cls, value: Any) -> Any:
        return None if value in (None, "") else parse_cny(value, field="budget_per_night_cny")


class WeatherForecastArgs(BaseModel):
    city: str | None = Field(None, max_length=32, description="城市；缺省取行程目的地")
    start_date: Date | None = Field(None, description="起始日期 YYYY-MM-DD；缺省取行程出发日期")
    days: int | None = Field(None, ge=1, le=7, description="预报天数 1-7；缺省按行程天数")

    @field_validator("start_date", mode="before")
    @classmethod
    def _date(cls, value: Any) -> Any:
        return _coerce_optional_date(value)


class LunarAlmanacArgs(BaseModel):
    date: Date | None = Field(None, description="查询日期 YYYY-MM-DD；缺省取行程出发日期")

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, value: Any) -> Any:
        return _coerce_optional_date(value)


class FlightSearchArgs(BaseModel):
    dep_city: str | None = Field(None, max_length=32, description="出发城市名或机场三字码，如「上海」/「SHA」")
    arr_city: str | None = Field(None, max_length=32, description="到达城市名或机场三字码，如「成都」/「CTU」")
    date: Date | None = Field(None, description="出发日期 YYYY-MM-DD；缺省取行程出发日期")

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, value: Any) -> Any:
        return _coerce_optional_date(value)


class IntercityTransportArgs(BaseModel):
    origin: str | None = Field(None, max_length=32, description="出发城市；缺省取行程出发地")
    destination: str | None = Field(None, max_length=32, description="到达城市；缺省取行程目的地")
    date: Date | None = Field(None, description="出发日期 YYYY-MM-DD；缺省取行程出发日期")
    include_flight: bool = Field(True, description="长途（默认 >800km）时是否自动查询航班")

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, value: Any) -> Any:
        return _coerce_optional_date(value)


# --------------------------------------------------------------------------- 公共辅助


def _hub(ctx: ToolContext) -> Any:
    if ctx.mcp is None:
        raise ToolError(
            ToolErrorCode.UPSTREAM_ERROR,
            "实时数据服务（MCP）未配置，车次/路线/酒店/天气/黄历/航班均不可用。",
            hint=(
                "向用户说明实时数据暂不可用，可用差标与知识库回答；"
                "不要编造车次、航班号、票价、天气或黄历。"
            ),
        )
    return ctx.mcp


def _live(ctx: ToolContext) -> LiveTravelData:
    return LiveTravelData(
        _hub(ctx),
        station_ttl_s=float(settings.mcp_station_code_ttl_seconds),
    )


def _tool_error(exc: MCPError) -> ToolError:
    message = str(exc)
    code = ToolErrorCode.TIMEOUT if "超时" in message or "timeout" in message.lower() else (
        ToolErrorCode.UPSTREAM_ERROR
    )
    return ToolError(
        code,
        f"实时数据查询失败：{message}",
        hint=(
            "这是外部数据服务故障，不是参数问题。告知用户该项实时数据暂不可用，"
            "不要重复调用同一工具，也不要编造车次、航班、票价、天气或黄历。"
        ),
    )


def _missing(field: str, label: str) -> ToolError:
    return ToolError(
        ToolErrorCode.MISSING_ARGUMENTS,
        f"缺少{label}，无法查询实时数据。",
        field_issues=[
            FieldIssue(field=field, problem=f"缺少{label}", expected=f"用户明确给出的{label}", example="上海")
        ],
        hint=f"请先向用户询问{label}，不要猜测。",
    )


def _resolve(ctx: ToolContext, value: str | None, slot: str, field: str, label: str) -> str:
    text = str(value or "").strip() or str(ctx.draft.value(slot) or "").strip()
    if not text:
        raise _missing(field, label)
    return text


def _resolve_date(ctx: ToolContext, value: Date | None, field: str = "date") -> Date:
    if value is not None:
        return value
    raw = ctx.draft.value("depart_date")
    if raw:
        return parse_cn_date(raw, field=field)
    raise _missing(field, "出发日期")


def _trip_nights(ctx: ToolContext) -> int:
    depart_raw = ctx.draft.value("depart_date")
    return_raw = ctx.draft.value("return_date")
    if not depart_raw:
        return 1
    try:
        depart = parse_cn_date(depart_raw, field="depart_date")
        ret = parse_cn_date(return_raw, field="return_date") if return_raw else depart + timedelta(days=1)
    except ToolError:
        return 1
    return max(1, (ret - depart).days)


def _trip_days(ctx: ToolContext) -> int:
    depart_raw = ctx.draft.value("depart_date")
    return_raw = ctx.draft.value("return_date")
    if depart_raw and return_raw:
        try:
            depart = parse_cn_date(depart_raw, field="depart_date")
            ret = parse_cn_date(return_raw, field="return_date")
            return max(1, min(7, (ret - depart).days + 1))
        except ToolError:
            return 3
    return 3


def _money(value: float | None) -> str:
    if value is None:
        return "未提供"
    return f"{value:.0f}" if abs(value - int(value)) < 0.01 else f"{value:.2f}"


def _km(value: float | None) -> str:
    if value is None:
        return "未知"
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def _duration(minutes: int | None) -> str:
    if minutes is None:
        return "未知"
    hours, mins = divmod(int(round(minutes)), 60)
    if hours and mins:
        return f"{hours} 小时 {mins} 分"
    if hours:
        return f"{hours} 小时"
    return f"{mins} 分"


def _trim(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…（结果过长，已截断）"


def _driving_advice(route: DrivingRoute) -> str | None:
    if route.distance_km is None:
        return None
    if route.distance_km >= DRIVING_RECOMMEND_OTHER_KM:
        return (
            f"距离约 {_km(route.distance_km)} km，自驾需 {_duration(route.duration_minutes)}，"
            "强烈建议改用高铁或飞机"
        )
    if route.distance_km >= DRIVING_COMFORT_KM:
        return (
            f"距离约 {_km(route.distance_km)} km，自驾需 {_duration(route.duration_minutes)}，"
            "注意疲劳驾驶"
        )
    return None


# --------------------------------------------------------------------------- 渲染


def _render_train(tickets: TrainTickets, *, limit: int = TRAIN_TEXT_LIMIT) -> str:
    header = f"🚄 {tickets.summary}"
    if tickets.stations_resolved:
        header += "｜站点代码已自动解析"
    lines = [header]
    if tickets.notes:
        lines.extend(f"· {note}" for note in tickets.notes)
    lines.append(_trim(tickets.text, limit) or "（12306 未返回车次内容）")
    return "\n".join(lines)


def _render_driving(route: DrivingRoute) -> str:
    lines = [f"🚗 自驾 {route.origin}→{route.destination}"]
    if route.distance_km is not None:
        line = f"距离 {_km(route.distance_km)} km｜约 {_duration(route.duration_minutes)}"
        if route.tolls_cny is not None:
            line += f"｜过路费 {_money(route.tolls_cny)} 元"
        lines.append(line)
    else:
        lines.append("未能获得距离信息")
    if route.estimated:
        lines.append("⚠️ 以上为估算值（高德服务不可用，按城市直线距离 ×1.25 推算），过路费与实时路况需以高德为准")
    advice = _driving_advice(route)
    if advice:
        lines.append(f"⚠️ {advice}")
    lines.extend(f"· {note}" for note in route.notes)
    return "\n".join(lines)


def _render_hotels(search: HotelSearch) -> str:
    lines = [f"🏨 {search.city} 住宿搜索（关键词：{'、'.join(search.keywords)}）"]
    if search.budget_per_night is not None:
        lines.append(f"每晚上限：{_money(search.budget_per_night)} 元")
    if search.hits:
        for index, hit in enumerate(search.hits[:8], start=1):
            price = f"｜约 {_money(hit.cost_cny)} 元/晚" if hit.cost_cny is not None else "｜价格未提供"
            address = f"｜{hit.address}" if hit.address else ""
            lines.append(f"{index}. {hit.name}{price}{address}")
        if search.over_budget:
            lines.append(f"（已按预算过滤 {search.over_budget} 家超预算酒店）")
        if not search.structured:
            lines.append("（价格来自文本解析，仅供参考，预订前请核实）")
    else:
        lines.append("未解析出结构化酒店列表，原始返回如下：")
        lines.append(_trim(search.text, RAW_TEXT_LIMIT) or "（高德未返回酒店结果）")
    lines.extend(f"· {note}" for note in search.notes)
    return "\n".join(lines)


def _render_weather(report: WeatherReport) -> str:
    lines = [f"☀️ {report.city} 天气预报"]
    if report.structured:
        for day in report.days:
            temps = ""
            if day.night_temp or day.day_temp:
                temps = f"｜{day.night_temp or '?'}~{day.day_temp or '?'}°C"
            night = f"转{day.night_weather}" if day.night_weather and day.night_weather != day.day_weather else ""
            wind = f"｜{day.wind}" if day.wind else ""
            lines.append(f"  {day.date}：{day.day_weather or '未知'}{night}{temps}{wind}")
        if report.missing_dates:
            range_text = (
                f"（该服务当前提供 {report.available_from} ~ {report.available_to} 的预报）"
                if report.available_from
                else ""
            )
            lines.append(
                f"⚠️ {', '.join(report.missing_dates)} 暂无预报{range_text}，建议出行前 1-2 天再查"
            )
    else:
        lines.append("当前服务未返回结构化预报，原始数据如下：")
        lines.append(_trim(report.text, RAW_TEXT_LIMIT) or "（高德未返回天气结果）")
    return "\n".join(lines)


def _render_almanac(almanac: Almanac) -> str:
    if not any((almanac.lunar_date, almanac.ganzhi, almanac.yi, almanac.ji)):
        return "📅 黄历查询结果：\n" + (_trim(almanac.text, RAW_TEXT_LIMIT) or "（八字服务未返回内容）")
    lines = [f"📅 {almanac.solar_date}"]
    if almanac.lunar_date:
        lines.append(f"🌕 农历：{almanac.lunar_date}")
    if almanac.ganzhi:
        lines.append(f"🎯 干支：{almanac.ganzhi}")
    if almanac.yi:
        lines.append(f"👍 宜：{almanac.yi.replace(',', '、').replace('，', '、')}")
    if almanac.ji:
        lines.append(f"⛔ 忌：{almanac.ji.replace(',', '、').replace('，', '、')}")
    if almanac.auspicious is True:
        lines.append("✅ 今日宜出行")
    elif almanac.auspicious is False:
        lines.append("⚠️ 今日忌出行，如无法调整请多注意安全")
    else:
        lines.append("🟡 今日对出行无特别宜忌，可正常安排")
    lines.append("📍 黄历仅供参考，行程请按实际情况安排")
    return "\n".join(lines)


def _render_flight(flight: FlightSearch) -> str:
    lines = [f"✈️ 航班 {flight.dep_city}({flight.dep_code})→{flight.arr_city}({flight.arr_code}) {flight.date}"]
    lines.append(_trim(flight.text, RAW_TEXT_LIMIT) or "（航班服务未返回内容）")
    return "\n".join(lines)


def _recommend_transport(
    distance_km: float | None,
    has_train: bool,
    driving: DrivingRoute | None,
    flight: FlightSearch | None,
    flight_skipped_reason: str,
) -> tuple[str, str]:
    """给出确定性的交通方式建议。规则可单测，不交给模型自由发挥。"""
    if distance_km is not None and distance_km >= settings.flight_auto_trigger_km:
        if flight is not None:
            return "flight", f"距离约 {_km(distance_km)} km，超过 {settings.flight_auto_trigger_km:.0f} km，飞机最省时"
        if has_train:
            return "train", "距离较远但未查到航班，优先高铁/火车"
        return "driving", "未能查到航班与车次，自驾为兜底方案但耗时较长"
    if has_train:
        return "train", "高铁/火车班次稳定、受天气影响小，综合最优"
    if driving is not None and (driving.distance_km or 0) < DRIVING_RECOMMEND_OTHER_KM:
        return "driving", "距离适中，自驾灵活"
    if flight is not None:
        return "flight", "长途出行，飞机更省时"
    if driving is not None:
        return "driving", "其它方式不可用，自驾为兜底方案"
    return "unknown", f"没有可用的交通数据（{flight_skipped_reason}）"


_TRANSPORT_LABELS = {"train": "高铁/火车", "driving": "自驾", "flight": "飞机", "unknown": "待定"}


# --------------------------------------------------------------------------- 只读工具


async def _query_train_tickets(args: TrainTicketsArgs, ctx: ToolContext) -> ToolReturn:
    origin = _resolve(ctx, args.origin, "origin_city", "origin", "出发城市")
    destination = _resolve(ctx, args.destination, "destination_city", "destination", "到达城市")
    travel_date = _resolve_date(ctx, args.date)
    data = _live(ctx)
    try:
        tickets = await data.train_tickets(origin, destination, travel_date.isoformat())
    except MCPError as exc:
        raise _tool_error(exc) from exc
    payload = {
        "origin": tickets.origin,
        "destination": tickets.destination,
        "date": tickets.date,
        "from_station_code": tickets.from_code,
        "to_station_code": tickets.to_code,
        "stations_resolved": tickets.stations_resolved,
        "raw": _trim(tickets.text, COMPOSITE_TEXT_LIMIT),
        "notes": tickets.notes,
    }
    return ToolReturn(
        data={"train": payload},
        rendered=_render_train(tickets),
        draft_updates={"origin_city": origin, "destination_city": destination, "depart_date": travel_date},
    )


async def _plan_driving_route(args: DrivingRouteArgs, ctx: ToolContext) -> ToolReturn:
    origin = _resolve(ctx, args.origin, "origin_city", "origin", "出发城市")
    destination = _resolve(ctx, args.destination, "destination_city", "destination", "到达城市")
    data = _live(ctx)
    try:
        route = await data.driving_route(origin, destination)
    except MCPError as exc:
        raise _tool_error(exc) from exc
    payload = {
        "origin": route.origin,
        "destination": route.destination,
        "distance_km": route.distance_km,
        "duration_minutes": route.duration_minutes,
        "tolls_cny": route.tolls_cny,
        "estimated": route.estimated,
        "notes": route.notes,
    }
    return ToolReturn(data={"driving": payload}, rendered=_render_driving(route))


async def _search_local_hotels(args: LocalHotelSearchArgs, ctx: ToolContext) -> ToolReturn:
    city = _resolve(ctx, args.city, "destination_city", "city", "住宿城市")
    night_budget: float | None = None
    if args.budget_per_night_cny is not None:
        night_budget = float(args.budget_per_night_cny)
    else:
        total_raw = ctx.draft.value("budget_cny")
        if total_raw not in (None, ""):
            night_budget = float(Decimal(str(total_raw)) / _trip_nights(ctx))
            if night_budget <= 0:
                # 预算 0 更可能是"用户没填/占位"，不能据此把酒店全过滤成"超预算"。
                night_budget = None

    keywords: list[str] = []
    if args.keywords:
        keywords = [args.keywords]
    elif args.star:
        keywords = [f"{city} {args.star}星级酒店", f"{city} 酒店"]

    data = _live(ctx)
    try:
        search = await data.hotel_search(city, budget_per_night=night_budget, keywords=keywords)
    except MCPError as exc:
        raise _tool_error(exc) from exc

    payload = {
        "city": search.city,
        "budget_per_night_cny": search.budget_per_night,
        "keywords": search.keywords,
        "over_budget": search.over_budget,
        "hotels": [
            {"name": hit.name, "cost_cny": hit.cost_cny, "address": hit.address}
            for hit in search.hits[:8]
        ],
        "structured": search.structured,
    }
    return ToolReturn(
        data={"hotels": payload},
        rendered=_render_hotels(search),
        draft_attributes={"last_hotels": payload},
    )


async def _query_weather_forecast(args: WeatherForecastArgs, ctx: ToolContext) -> ToolReturn:
    city = _resolve(ctx, args.city, "destination_city", "city", "目的地城市")
    start: Date | None = args.start_date
    if start is None:
        raw = ctx.draft.value("depart_date")
        start = parse_cn_date(raw, field="depart_date") if raw else None
    days = args.days or _trip_days(ctx)
    data = _live(ctx)
    try:
        report = await data.weather(city, start=start, days=days)
    except MCPError as exc:
        raise _tool_error(exc) from exc

    payload = {
        "city": report.city,
        "start_date": start.isoformat() if start else None,
        "days_requested": days,
        "days": [
            {
                "date": day.date,
                "day_weather": day.day_weather,
                "night_weather": day.night_weather,
                "day_temp": day.day_temp,
                "night_temp": day.night_temp,
                "wind": day.wind,
            }
            for day in report.days
        ],
        "available_from": report.available_from,
        "available_to": report.available_to,
        "missing_dates": report.missing_dates,
        "structured": report.structured,
    }
    return ToolReturn(
        data={"weather": payload},
        rendered=_render_weather(report),
        draft_attributes={"last_weather": payload},
    )


async def _query_lunar_almanac(args: LunarAlmanacArgs, ctx: ToolContext) -> ToolReturn:
    day = _resolve_date(ctx, args.date)
    data = _live(ctx)
    try:
        almanac = await data.almanac(day)
    except MCPError as exc:
        raise _tool_error(exc) from exc

    payload = {
        "date": almanac.solar_date or day.isoformat(),
        "lunar_date": almanac.lunar_date,
        "ganzhi": almanac.ganzhi,
        "yi": almanac.yi,
        "ji": almanac.ji,
        "auspicious_for_travel": almanac.auspicious,
    }
    return ToolReturn(
        data={"almanac": payload},
        rendered=_render_almanac(almanac),
        draft_attributes={"last_almanac": payload},
    )


async def _query_flights(args: FlightSearchArgs, ctx: ToolContext) -> ToolReturn:
    dep_city = _resolve(ctx, args.dep_city, "origin_city", "dep_city", "出发城市")
    arr_city = _resolve(ctx, args.arr_city, "destination_city", "arr_city", "到达城市")
    travel_date = _resolve_date(ctx, args.date)
    data = _live(ctx)
    if not data.supports(FLIGHT_SERVER):
        raise ToolError(
            ToolErrorCode.UPSTREAM_ERROR,
            "航班查询服务未配置。",
            hint="告知用户当前只提供火车/自驾方案，可建议用户自行查询航班。",
        )
    try:
        flight = await data.flights(dep_city, arr_city, travel_date)
    except AirportCodeError as exc:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"无法把「{exc.city}」转换为机场三字码。",
            field_issues=[
                FieldIssue(
                    field="dep_city" if exc.city == dep_city else "arr_city",
                    problem="不是已知城市或 IATA 三字码",
                    expected="中文城市名（如「上海」）或 3 位机场代码（如「SHA」）",
                    example="上海",
                )
            ],
            hint="城市名请使用常见中文名，或直接给出机场三字码。",
        ) from exc
    except MCPError as exc:
        raise _tool_error(exc) from exc

    payload = {
        "dep_city": flight.dep_city,
        "arr_city": flight.arr_city,
        "dep_code": flight.dep_code,
        "arr_code": flight.arr_code,
        "date": flight.date,
        "raw": _trim(flight.text, COMPOSITE_TEXT_LIMIT),
    }
    return ToolReturn(
        data={"flight": payload},
        rendered=_render_flight(flight),
        draft_updates={"origin_city": dep_city, "destination_city": arr_city, "depart_date": travel_date},
    )


# --------------------------------------------------------------------------- 复合工具


async def _guard(coro: Any, seconds: float) -> tuple[Any | None, str | None]:
    """给子查询设独立预算：单项超时不能拖垮整次城际交通查询。"""
    try:
        return await asyncio.wait_for(coro, timeout=seconds), None
    except asyncio.TimeoutError:
        return None, f"查询超时（>{seconds:.0f}s）"
    except MCPError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - 子查询失败只影响该分项
        return None, f"{type(exc).__name__}: {exc}"


async def _plan_intercity_transport(args: IntercityTransportArgs, ctx: ToolContext) -> ToolReturn:
    origin = _resolve(ctx, args.origin, "origin_city", "origin", "出发城市")
    destination = _resolve(ctx, args.destination, "destination_city", "destination", "到达城市")
    travel_date = _resolve_date(ctx, args.date)
    data = _live(ctx)

    estimate = estimate_distance(origin, destination)
    distance_km = estimate.road_km if estimate is not None else None

    should_fly = bool(args.include_flight) and data.supports(FLIGHT_SERVER)
    if not args.include_flight:
        flight_skip_reason = "调用方关闭了航班查询"
    elif not data.supports(FLIGHT_SERVER):
        flight_skip_reason = "未配置航班 MCP 服务"
    elif distance_km is None:
        flight_skip_reason = "距离未知"
    elif distance_km >= settings.flight_auto_trigger_km:
        flight_skip_reason = ""
    else:
        should_fly = False
        flight_skip_reason = (
            f"距离约 {_km(distance_km)} km，未达 {settings.flight_auto_trigger_km:.0f} km 自动触发阈值"
        )

    # 三个查询互相独立，并发执行：串行会把"外部服务慢"直接叠成整轮超时，
    # 而并发下总耗时约等于最慢的一项，且单项超时不影响其它分项。
    train_coro = _guard(data.train_tickets(origin, destination, travel_date.isoformat()), 65.0)
    driving_coro = _guard(data.driving_route(origin, destination), 25.0)
    flight_coro = _guard(data.flights(origin, destination, travel_date), 25.0) if should_fly else None

    gathered = await asyncio.gather(
        train_coro, driving_coro, *([flight_coro] if flight_coro is not None else [])
    )
    train, train_error = gathered[0]
    driving, driving_error = gathered[1]
    flight: FlightSearch | None = None
    flight_error: str | None = None
    if flight_coro is not None:
        flight, flight_error = gathered[2]

    if train is None and driving is None and flight is None:
        raise ToolError(
            ToolErrorCode.UPSTREAM_ERROR,
            "城际交通查询全部失败："
            + "；".join(filter(None, [train_error, driving_error, flight_error]))[:400],
            hint="告知用户实时交通数据暂不可用，可先用差标/知识库回答，不要编造车次或航班。",
        )

    mode, reason = _recommend_transport(
        distance_km,
        has_train=train is not None,
        driving=driving,
        flight=flight,
        flight_skipped_reason=flight_skip_reason,
    )

    sections: list[str] = []
    header = f"🧭 {origin}→{destination}｜{travel_date.isoformat()}"
    if driving is not None and driving.distance_km is not None and not driving.estimated:
        header += f"｜距离约 {_km(driving.distance_km)} km"
    elif distance_km is not None:
        header += f"｜估算距离约 {_km(distance_km)} km"
    sections.append(header)
    data_payload: dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "date": travel_date.isoformat(),
        "distance_km": distance_km,
        "recommendation": {"mode": mode, "label": _TRANSPORT_LABELS[mode], "reason": reason},
    }

    if train is not None:
        sections.append("")
        sections.append(_render_train(train, limit=COMPOSITE_TEXT_LIMIT))
        data_payload["train"] = {
            "ok": True,
            "from_station_code": train.from_code,
            "to_station_code": train.to_code,
            "raw": _trim(train.text, COMPOSITE_TEXT_LIMIT),
        }
    else:
        sections.append("")
        sections.append(f"🚄 车次查询失败：{train_error}")
        data_payload["train"] = {"ok": False, "error": train_error}

    if driving is not None:
        sections.append("")
        sections.append(_render_driving(driving))
        data_payload["driving"] = {
            "distance_km": driving.distance_km,
            "duration_minutes": driving.duration_minutes,
            "tolls_cny": driving.tolls_cny,
            "estimated": driving.estimated,
        }
    else:
        sections.append("")
        sections.append(f"🚗 自驾路线查询失败：{driving_error}")
        data_payload["driving"] = {"ok": False, "error": driving_error}

    if flight is not None:
        sections.append("")
        sections.append(_render_flight(flight))
        data_payload["flight"] = {
            "dep_code": flight.dep_code,
            "arr_code": flight.arr_code,
            "raw": _trim(flight.text, COMPOSITE_TEXT_LIMIT),
        }
    elif should_fly:
        sections.append("")
        sections.append(f"✈️ 航班查询失败：{flight_error}")
        data_payload["flight"] = {"ok": False, "error": flight_error}
    else:
        data_payload["flight"] = {"ok": False, "skipped": True, "reason": flight_skip_reason}

    sections.append("")
    sections.append(f"🧭 建议：{_TRANSPORT_LABELS[mode]}（{reason}）")
    if estimate is not None and (driving is None or driving.estimated):
        sections.append("· 距离为城市中心点估算，真实里程以高德/12306 返回为准")

    return ToolReturn(
        data={"transport": data_payload},
        rendered="\n".join(sections),
        draft_attributes={
            "last_transport": {
                "origin": origin,
                "destination": destination,
                "date": travel_date.isoformat(),
                "distance_km": distance_km,
                "mode": mode,
                "reason": reason,
                "train_ok": train is not None,
                "flight_ok": flight is not None,
                "driving_km": driving.distance_km if driving else None,
            }
        },
    )


# --------------------------------------------------------------------------- 注册


def register_mcp_tools(registry: ToolRegistry) -> int:
    """注册实时数据工具。仅在配置了 MCP 服务时调用，避免模型看到必然失败的工具。"""
    specs = [
        ToolSpec(
            name="plan_intercity_transport",
            description=(
                "查询两地之间的完整交通方案：12306 车次（自动解析站点代码）+ 高德自驾（距离/时间/过路费）"
                "+ 长途自动查航班，并给出建议。规划出差行程时优先调用这个复合工具。只读。"
            ),
            args_model=IntercityTransportArgs,
            handler=_plan_intercity_transport,
            access=ToolAccess.READ,
            timeout_seconds=100.0,
        ),
        ToolSpec(
            name="query_train_tickets",
            description=(
                "查询 12306 火车票车次与时刻表，自动把城市名解析成车站代码。"
                "需要车次/票价/余票细节或用户只要火车方案时使用。只读。"
            ),
            args_model=TrainTicketsArgs,
            handler=_query_train_tickets,
            access=ToolAccess.READ,
            timeout_seconds=100.0,
        ),
        ToolSpec(
            name="plan_driving_route",
            description=(
                "查询高德自驾路线：距离、预计耗时、过路费，并提示是否适合自驾。只读。"
            ),
            args_model=DrivingRouteArgs,
            handler=_plan_driving_route,
            access=ToolAccess.READ,
            timeout_seconds=60.0,
        ),
        ToolSpec(
            name="search_local_hotels",
            description=(
                "用高德地图搜索目的地酒店/民宿，按每晚预算自动生成关键词并过滤超预算结果。"
                "返回的是实时 POI 参考，不是可预订报价；报价请用 search_hotels。只读。"
            ),
            args_model=LocalHotelSearchArgs,
            handler=_search_local_hotels,
            access=ToolAccess.READ,
            timeout_seconds=60.0,
        ),
        ToolSpec(
            name="query_weather_forecast",
            description=(
                "查询目的地多日天气预报（高德），按行程日期逐日匹配，超出预报范围会明确说明。只读。"
            ),
            args_model=WeatherForecastArgs,
            handler=_query_weather_forecast,
            access=ToolAccess.READ,
            timeout_seconds=45.0,
        ),
        ToolSpec(
            name="query_lunar_almanac",
            description=(
                "查询指定日期的农历、干支、宜忌与是否宜出行（八字黄历）。"
                "黄历仅作传统文化参考，不影响差标与预订决策。只读。"
            ),
            args_model=LunarAlmanacArgs,
            handler=_query_lunar_almanac,
            access=ToolAccess.READ,
            timeout_seconds=45.0,
        ),
        ToolSpec(
            name="query_flights",
            description=(
                "查询航班时刻（城市名自动转机场三字码）。用户明确要航班、或距离超过 "
                f"{settings.flight_auto_trigger_km:.0f} km 需要对比飞机方案时使用。只读。"
            ),
            args_model=FlightSearchArgs,
            handler=_query_flights,
            access=ToolAccess.READ,
            timeout_seconds=60.0,
        ),
    ]
    for spec in specs:
        registry.register(spec)
    logger.info("mcp_tools.registered", tools=[spec.name for spec in specs])
    return len(specs)


__all__ = [
    "COMPOSITE_TEXT_LIMIT",
    "TRAIN_TEXT_LIMIT",
    "register_mcp_tools",
]
