"""MCP 工具层测试：参数回填、预算筛选、自动触发航班、错误降级与注册边界。

工具层是"协议细节"与"业务语义"的接缝：这里断言的是接缝处的行为——
模型不填参数时用草稿回填、MCP 挂了给可读错误、长途自动补航班，而不是再去测一遍解析。
"""

from __future__ import annotations

from datetime import date

from app.agent.travel_orchestrator import TravelOrchestrator
from app.core.memory.profile import ClaimsProfileProvider
from app.core.memory.slots import SlotSource
from app.core.memory.trip_draft import TripDraft
from app.core.tools.errors import ToolErrorCode, ToolFailure, ToolSuccess
from app.core.tools.spec import ToolContext
from app.core.tools.travel_builtin import build_travel_registry
from app.domain.travel.policy import default_travel_policy
from app.infrastructure.mcp.hub import (
    FLIGHT_SERVER,
    GAODE_SERVER,
    TRAIN_SERVER,
    MCPError,
)
from tests.conftest import COMPLETE_TRIP_SLOTS, make_travel_principal
from tests.mcp_fakes import FakeMCPHub, default_handlers

MCP_TOOL_NAMES = {
    "plan_intercity_transport",
    "query_train_tickets",
    "plan_driving_route",
    "search_local_hotels",
    "query_weather_forecast",
    "query_lunar_almanac",
    "query_flights",
}


async def make_ctx(hub: FakeMCPHub | None, **slot_overrides) -> ToolContext:
    principal = make_travel_principal()
    profile = await ClaimsProfileProvider().get(principal)
    assert profile is not None
    draft = TripDraft(session_id="s-mcp", tenant_id=principal.tenant_id, subject_id=principal.subject_id)
    slots = {
        **COMPLETE_TRIP_SLOTS,
        "origin_city": "上海",
        "destination_city": "杭州",
        "depart_date": "2026-11-20",
        "hotel_needed": True,
    }
    slots.update(slot_overrides)
    draft.merge(slots, source=SlotSource.USER, turn=1)
    return ToolContext(
        principal=principal,
        profile=profile,
        draft=draft,
        policy=default_travel_policy(version="test-v1"),
        session_id="s-mcp",
        trace_id="t-mcp",
        today=date(2026, 11, 13),
        mcp=hub,
    )


def _hub(**overrides) -> FakeMCPHub:
    handlers = default_handlers()
    handlers.update(overrides)
    return FakeMCPHub(handlers)


def _registry():
    return build_travel_registry(include_mcp=True)


# --------------------------------------------------------------------------- 注册边界


def test_mcp_tools_are_optional() -> None:
    assert MCP_TOOL_NAMES.isdisjoint(set(build_travel_registry().names()))
    names = set(build_travel_registry(include_mcp=True).names())
    assert MCP_TOOL_NAMES <= names
    assert len(names) == len(set(build_travel_registry().names())) + len(MCP_TOOL_NAMES)


def test_mcp_tools_are_read_only() -> None:
    catalog = {item["name"]: item for item in _registry().mcp_tool_catalog()}
    for name in MCP_TOOL_NAMES:
        assert catalog[name]["annotations"]["access"] == "read"
        assert catalog[name]["annotations"]["requiresConfirmation"] is False


def test_mcp_tools_export_function_schema() -> None:
    definitions = {d["function"]["name"]: d for d in _registry().function_definitions()}
    for name in MCP_TOOL_NAMES:
        assert definitions[name]["function"]["parameters"]["type"] == "object"


# --------------------------------------------------------------------------- 参数与回填


async def test_train_tool_uses_draft_slots() -> None:
    ctx = await make_ctx(_hub())
    result = await _registry().invoke("query_train_tickets", {}, ctx)

    assert isinstance(result, ToolSuccess), result
    assert result.data["train"]["from_station_code"] == "SHH"
    assert result.data["train"]["to_station_code"] == "HZH"
    assert "G7301" in result.rendered


async def test_train_tool_reports_missing_fields() -> None:
    ctx = await make_ctx(_hub())
    ctx.draft = TripDraft(session_id="empty")
    result = await _registry().invoke("query_train_tickets", {}, ctx)

    assert isinstance(result, ToolFailure)
    assert result.code is ToolErrorCode.MISSING_ARGUMENTS
    assert result.field_issues[0].field == "origin"
    assert "出发城市" in result.message


async def test_train_tool_translates_mcp_failure() -> None:
    hub = FakeMCPHub(default_handlers(), fail_servers={TRAIN_SERVER})
    ctx = await make_ctx(hub)
    result = await _registry().invoke("query_train_tickets", {}, ctx)

    assert isinstance(result, ToolFailure)
    assert result.code is ToolErrorCode.UPSTREAM_ERROR
    assert "实时数据查询失败" in result.message
    assert "不要编造" in (result.hint or "")


async def test_mcp_tool_without_hub_is_explicit() -> None:
    ctx = await make_ctx(None)
    result = await _registry().invoke("query_weather_forecast", {}, ctx)

    assert isinstance(result, ToolFailure)
    assert result.code is ToolErrorCode.UPSTREAM_ERROR
    assert "未配置" in result.message


# --------------------------------------------------------------------------- 各工具


async def test_driving_tool_renders_distance_time_tolls() -> None:
    ctx = await make_ctx(_hub())
    result = await _registry().invoke("plan_driving_route", {}, ctx)

    assert isinstance(result, ToolSuccess), result
    assert result.data["driving"]["distance_km"] == 176.4
    assert result.data["driving"]["duration_minutes"] == 140
    assert result.data["driving"]["tolls_cny"] == 85
    assert "过路费 85 元" in result.rendered
    assert "自驾" in result.rendered


async def test_hotel_tool_derives_budget_from_draft() -> None:
    # 总预算 3000 / 2 晚 = 每晚上限 1500 → 关键词应升级为「高端酒店」。
    ctx = await make_ctx(_hub(), budget_cny="3000", return_date="2026-11-22")
    result = await _registry().invoke("search_local_hotels", {}, ctx)

    assert isinstance(result, ToolSuccess), result
    assert result.data["hotels"]["budget_per_night_cny"] == 1500.0
    assert result.data["hotels"]["keywords"][0] == "杭州 高端酒店"
    assert "每晚上限：1500 元" in result.rendered
    assert result.data["hotels"]["over_budget"] == 0
    assert ctx.draft.last_hotels is not None
    assert ctx.draft.last_hotels["budget_per_night_cny"] == 1500.0


async def test_hotel_tool_explicit_budget_filters_results() -> None:
    ctx = await make_ctx(_hub())
    result = await _registry().invoke("search_local_hotels", {"budget_per_night_cny": "400"}, ctx)

    assert isinstance(result, ToolSuccess), result
    names = [hit["name"] for hit in result.data["hotels"]["hotels"]]
    assert names == ["甲酒店（西湖店）", "丙民宿"]
    assert result.data["hotels"]["over_budget"] == 1


async def test_hotel_tool_reports_service_failure() -> None:
    hub = FakeMCPHub(default_handlers(), fail_servers={GAODE_SERVER})
    ctx = await make_ctx(hub)
    result = await _registry().invoke("search_local_hotels", {}, ctx)

    assert isinstance(result, ToolFailure)
    assert result.code is ToolErrorCode.UPSTREAM_ERROR
    assert "酒店搜索失败" in result.message


async def test_hotel_tool_ignores_zero_budget() -> None:
    # 预算 0 更可能是"没填"，不能据此把所有酒店判成超预算。
    ctx = await make_ctx(_hub(), budget_cny="0")
    result = await _registry().invoke("search_local_hotels", {}, ctx)

    assert isinstance(result, ToolSuccess), result
    assert result.data["hotels"]["budget_per_night_cny"] is None
    assert result.data["hotels"]["over_budget"] == 0


async def test_weather_tool_renders_multi_day_and_missing_dates() -> None:
    ctx = await make_ctx(_hub())
    result = await _registry().invoke("query_weather_forecast", {}, ctx)

    assert isinstance(result, ToolSuccess), result
    assert "2026-11-20" in result.rendered
    assert "2026-11-21" in result.rendered
    assert "2026-11-22 暂无预报" in result.rendered
    assert result.data["weather"]["missing_dates"] == ["2026-11-22"]
    assert ctx.draft.last_weather is not None


async def test_almanac_tool_renders_verdict() -> None:
    ctx = await make_ctx(_hub())
    result = await _registry().invoke("query_lunar_almanac", {}, ctx)

    assert isinstance(result, ToolSuccess), result
    assert "宜出行" in result.rendered
    assert "十月初一" in result.rendered
    assert result.data["almanac"]["auspicious_for_travel"] is True
    assert ctx.draft.last_almanac is not None


async def test_flight_tool_converts_city_to_iata() -> None:
    ctx = await make_ctx(_hub())
    result = await _registry().invoke(
        "query_flights", {"dep_city": "上海", "arr_city": "成都", "date": "2026-11-20"}, ctx
    )

    assert isinstance(result, ToolSuccess), result
    assert result.data["flight"]["dep_code"] == "SHA"
    assert result.data["flight"]["arr_code"] == "CTU"
    assert "MU5401" in result.rendered


async def test_flight_tool_unknown_city_is_field_error() -> None:
    ctx = await make_ctx(_hub())
    result = await _registry().invoke("query_flights", {"dep_city": "火星", "arr_city": "成都"}, ctx)

    assert isinstance(result, ToolFailure)
    assert result.code is ToolErrorCode.INVALID_ARGUMENTS
    assert result.field_issues[0].field == "dep_city"


# --------------------------------------------------------------------------- 城际交通复合工具


async def test_composite_auto_queries_flight_over_800km() -> None:
    ctx = await make_ctx(_hub(), destination_city="成都")
    result = await _registry().invoke("plan_intercity_transport", {}, ctx)

    assert isinstance(result, ToolSuccess), result
    transport = result.data["transport"]
    assert transport["distance_km"] is not None and transport["distance_km"] > 800
    assert transport["recommendation"]["mode"] == "flight"
    assert "✈️" in result.rendered
    assert "MU5401" in result.rendered
    assert ctx.draft.last_transport is not None
    assert ctx.draft.last_transport["flight_ok"] is True


async def test_composite_skips_flight_under_threshold() -> None:
    hub = _hub()
    ctx = await make_ctx(hub)
    result = await _registry().invoke("plan_intercity_transport", {}, ctx)

    assert isinstance(result, ToolSuccess), result
    assert hub.called(FLIGHT_SERVER, "searchFlightsByDepArr") == []
    flight = result.data["transport"]["flight"]
    assert flight["skipped"] is True
    assert "未达" in flight["reason"]
    assert "建议：高铁/火车" in result.rendered


async def test_composite_survives_partial_failure() -> None:
    hub = FakeMCPHub(default_handlers(), fail_servers={TRAIN_SERVER})
    ctx = await make_ctx(hub)
    result = await _registry().invoke("plan_intercity_transport", {}, ctx)

    assert isinstance(result, ToolSuccess), result
    assert result.data["transport"]["train"]["ok"] is False
    assert result.data["transport"]["driving"]["distance_km"] == 176.4
    assert "车次查询失败" in result.rendered
    assert result.data["transport"]["recommendation"]["mode"] == "driving"


async def test_composite_all_fail_reports_upstream_error() -> None:
    handlers = default_handlers()
    handlers[(TRAIN_SERVER, "get-tickets")] = lambda _args: MCPError("12306 不可用")
    hub = FakeMCPHub(handlers, servers=(TRAIN_SERVER, GAODE_SERVER), fail_servers={GAODE_SERVER})
    ctx = await make_ctx(hub)
    result = await _registry().invoke(
        "plan_intercity_transport", {"origin": "甲城", "destination": "乙城"}, ctx
    )

    assert isinstance(result, ToolFailure)
    assert result.code is ToolErrorCode.UPSTREAM_ERROR
    assert "城际交通查询全部失败" in result.message


# --------------------------------------------------------------------------- 编排接入


def _draft(**slots) -> TripDraft:
    draft = TripDraft(session_id="s1")
    draft.merge(
        {
            "origin_city": "上海",
            "destination_city": "杭州",
            "depart_date": "2026-11-20",
            "purpose": "client",
            **slots,
        },
        source=SlotSource.USER,
        turn=1,
    )
    return draft


def test_realtime_steps_added_for_complete_trip() -> None:
    draft = _draft(hotel_needed=True)
    steps = TravelOrchestrator._realtime_steps(draft)
    assert [name for name, _ in steps] == [
        "plan_intercity_transport",
        "search_local_hotels",
        "query_weather_forecast",
        "query_lunar_almanac",
    ]
    assert steps[0][1]["date"] == "2026-11-20"


def test_realtime_steps_empty_without_depart_date() -> None:
    draft = TripDraft(session_id="s1")
    draft.merge({"origin_city": "上海", "destination_city": "杭州"}, source=SlotSource.USER, turn=1)
    assert TravelOrchestrator._realtime_steps(draft) == []


async def test_plan_steps_include_realtime_only_with_hub() -> None:
    orchestrator = object.__new__(TravelOrchestrator)  # 只调纯函数，不需要完整依赖
    ctx = await make_ctx(_hub())
    with_hub = [name for name, _ in orchestrator._plan_steps(ctx)]
    assert "plan_intercity_transport" in with_hub
    assert with_hub[-4:] == [
        "plan_intercity_transport",
        "search_local_hotels",
        "query_weather_forecast",
        "query_lunar_almanac",
    ]

    ctx.mcp = None
    without_hub = [name for name, _ in orchestrator._plan_steps(ctx)]
    assert "plan_intercity_transport" not in without_hub
    assert without_hub[-2:] == ["check_travel_compliance", "plan_trip"]


async def test_realtime_lines_summarize_snapshots() -> None:
    ctx = await make_ctx(_hub())
    ctx.draft.last_transport = {
        "origin": "上海",
        "destination": "成都",
        "date": "2026-11-20",
        "distance_km": 2069.4,
        "mode": "flight",
        "train_ok": True,
        "flight_ok": True,
        "driving_km": 2586.7,
    }
    ctx.draft.last_weather = {
        "city": "成都",
        "days": [{"date": "2026-11-20", "day_weather": "多云", "day_temp": "18", "night_temp": "11"}],
    }
    ctx.draft.last_almanac = {"date": "2026-11-20", "auspicious_for_travel": True}
    ctx.draft.last_hotels = {"city": "成都", "hotels": [{"name": "甲"}], "budget_per_night_cny": 600}

    lines = TravelOrchestrator._realtime_lines(ctx)
    joined = "\n".join(lines)
    assert "上海→成都" in joined
    assert "建议 飞机" in joined
    assert "多云" in joined
    assert "宜出行" in joined
    assert "酒店候选 1 家" in joined


async def test_plan_execute_runs_realtime_tools_end_to_end() -> None:
    """完整跑一遍 PLAN_EXECUTE：方案生成后自动补车次/天气/黄历，再交给模型汇总。"""
    from app.agent.travel_orchestrator import RunRequest
    from tests.conftest import FakeLLM, ScriptedTurn, make_travel_harness, seed_trip

    hub = _hub()
    harness = make_travel_harness(FakeLLM([ScriptedTurn(content="已按方案安排。")]))
    harness.orchestrator._mcp = hub
    harness.orchestrator._registry = build_travel_registry(include_mcp=True)

    principal = make_travel_principal()
    await seed_trip(
        harness.sessions,
        principal,
        session_id="s-plan-mcp",
        slots={
            **COMPLETE_TRIP_SLOTS,
            "destination_city": "杭州",
            "depart_date": "2026-11-20",
            "return_date": "2026-11-21",
            "hotel_needed": True,
            "budget_cny": "3000",
        },
    )

    outcome = await harness.orchestrator.run(
        RunRequest(
            principal=principal,
            user_text="帮我规划去杭州的出差行程，住一晚",
            session_id="s-plan-mcp",
        )
    )

    assert outcome.paradigm == "plan_execute"
    names = [call.name for call in outcome.tool_calls]
    assert "plan_intercity_transport" in names
    assert "query_weather_forecast" in names
    assert "query_lunar_almanac" in names
    assert all(call.ok for call in outcome.tool_calls if call.name in MCP_TOOL_NAMES)
    assert outcome.text == "已按方案安排。"

    state = await harness.sessions.load("s-plan-mcp", tenant_id=principal.tenant_id)
    assert state is not None
    assert state.draft.last_transport is not None
    assert state.draft.last_weather is not None
    assert state.draft.last_almanac is not None
