"""差旅编排器端到端：路由分发、确认门、二次核价、记忆注入、降级。"""

from __future__ import annotations

from decimal import Decimal

import pytest
from fastapi import HTTPException

from app.agent.travel_orchestrator import RunRequest
from app.domain.schemas import ConfirmActionRequest, ConfirmDecision
from tests.conftest import (
    FakeLLM,
    ScriptedToolCall,
    ScriptedTurn,
    make_travel_harness,
    make_travel_principal,
    seed_planned_trip,
    seed_trip,
)


async def test_booking_is_gated_and_only_confirm_executes() -> None:
    llm = FakeLLM([ScriptedTurn(tool_calls=[ScriptedToolCall("book_trip", {})])])
    h = make_travel_harness(llm)
    principal = make_travel_principal()
    _, plan = await seed_planned_trip(h, principal, session_id="s-book")

    outcome = await h.orchestrator.run(
        RunRequest(principal=principal, user_text="帮我订这个方案", session_id="s-book")
    )
    assert outcome.finish_reason == "confirmation_required"
    assert outcome.pending_action is not None
    assert plan["plan_id"] in outcome.pending_action.preview
    # 关键：写操作没有被执行。
    assert not h.supplier._bookings

    response = await h.orchestrator.confirm_action(
        principal,
        ConfirmActionRequest(action_id=outcome.pending_action.action_id, decision=ConfirmDecision.CONFIRM),
        trace_id="t-confirm",
    )
    assert response.status == "confirmed"
    assert response.result is not None
    assert len(response.result["bookings"]) == len(plan["segments"])
    assert h.supplier._bookings


async def test_price_change_aborts_confirmation() -> None:
    llm = FakeLLM([ScriptedTurn(tool_calls=[ScriptedToolCall("book_trip", {})])])
    h = make_travel_harness(llm)
    principal = make_travel_principal()
    _, plan = await seed_planned_trip(h, principal, session_id="s-price")

    outcome = await h.orchestrator.run(
        RunRequest(principal=principal, user_text="帮我订这个方案", session_id="s-price")
    )
    assert outcome.pending_action is not None

    first = plan["segments"][0]
    h.supplier.simulate_price_change(
        first["offer_id"], price_cny=Decimal(str(first["price_cny"])) * Decimal("1.10")
    )

    with pytest.raises(HTTPException) as excinfo:
        await h.orchestrator.confirm_action(
            principal,
            ConfirmActionRequest(
                action_id=outcome.pending_action.action_id, decision=ConfirmDecision.CONFIRM
            ),
            trace_id="t-price",
        )
    assert excinfo.value.status_code == 409
    assert "报价已变化" in str(excinfo.value.detail)
    assert not h.supplier._bookings


async def test_rejecting_confirmation_cancels_nothing() -> None:
    llm = FakeLLM([ScriptedTurn(tool_calls=[ScriptedToolCall("book_trip", {})])])
    h = make_travel_harness(llm)
    principal = make_travel_principal()
    await seed_planned_trip(h, principal, session_id="s-reject")
    outcome = await h.orchestrator.run(
        RunRequest(principal=principal, user_text="帮我订这个方案", session_id="s-reject")
    )
    assert outcome.pending_action is not None
    response = await h.orchestrator.confirm_action(
        principal,
        ConfirmActionRequest(action_id=outcome.pending_action.action_id, decision=ConfirmDecision.REJECT),
        trace_id="t-reject",
    )
    assert response.status == "rejected"
    assert not h.supplier._bookings


async def test_low_confidence_triggers_clarification_without_llm() -> None:
    h = make_travel_harness(FakeLLM([]))
    outcome = await h.orchestrator.run(
        RunRequest(principal=make_travel_principal(), user_text="安排一下")
    )
    assert outcome.finish_reason == "clarification_required"
    assert "还需要以下信息" in outcome.text
    assert outcome.paradigm == "clarify"
    assert h.llm.calls == []


async def test_policy_question_goes_direct_and_calls_policy_tool() -> None:
    h = make_travel_harness(FakeLLM([ScriptedTurn(content="经理职级国内经济舱，北京住宿上限 800 元/晚。")]))
    outcome = await h.orchestrator.run(
        RunRequest(principal=make_travel_principal(), user_text="公司差旅住宿标准是多少")
    )
    assert outcome.paradigm == "direct"
    assert "get_travel_policy" in [tc.name for tc in outcome.tool_calls]
    assert "800" in outcome.text


async def test_complete_trip_planning_uses_plan_execute_pipeline() -> None:
    h = make_travel_harness(FakeLLM([ScriptedTurn(content="已生成方案，请确认。")]))
    principal = make_travel_principal()
    await seed_trip(h.sessions, principal, session_id="s-plan")

    outcome = await h.orchestrator.run(
        RunRequest(principal=principal, user_text="帮我安排一下行程", session_id="s-plan")
    )
    assert outcome.paradigm == "plan_execute"
    names = [tc.name for tc in outcome.tool_calls]
    assert names == [
        "get_travel_policy",
        "search_flights",
        "check_travel_compliance",
        "plan_trip",
    ]
    assert outcome.text == "已生成方案，请确认。"


async def test_long_term_preferences_are_injected_into_prompt() -> None:
    from tests.conftest import FakeLongTerm

    ltm = FakeLongTerm(["我习惯靠窗座位", "我偏好亚朵酒店"])
    h = make_travel_harness(
        FakeLLM([ScriptedTurn(content="已按你的偏好安排。")]), long_term=ltm
    )
    principal = make_travel_principal()
    await seed_trip(h.sessions, principal, session_id="s-memory")
    await h.orchestrator.run(
        RunRequest(principal=principal, user_text="帮我安排一下行程", session_id="s-memory")
    )
    system_prompt = h.llm.calls[-1][0]["content"]
    assert "<memory>" in system_prompt
    assert "靠窗" in system_prompt


async def test_preference_statement_is_written_to_long_term_memory() -> None:
    h = make_travel_harness(FakeLLM([]))
    await h.orchestrator.run(
        RunRequest(principal=make_travel_principal(), user_text="我习惯坐靠窗，以后都订亚朵")
    )
    assert any("靠窗" in text for text in h.long_term.written)


async def test_session_is_bound_to_subject() -> None:
    llm = FakeLLM([ScriptedTurn(content="ok"), ScriptedTurn(content="ok")])
    h = make_travel_harness(llm)
    owner = make_travel_principal(subject_id="E10086")
    outcome = await h.orchestrator.run(RunRequest(principal=owner, user_text="你好"))
    intruder = make_travel_principal(subject_id="E99999")
    with pytest.raises(HTTPException) as excinfo:
        await h.orchestrator.run(
            RunRequest(principal=intruder, user_text="给我看看", session_id=outcome.session_id)
        )
    assert excinfo.value.status_code == 403


async def test_llm_failure_returns_deterministic_degraded_reply() -> None:
    h = make_travel_harness(FakeLLM([], fail_with="upstream 503"))
    principal = make_travel_principal()
    await seed_trip(h.sessions, principal, session_id="s-degrade")
    outcome = await h.orchestrator.run(
        RunRequest(principal=principal, user_text="帮我安排一下行程", session_id="s-degrade")
    )
    assert outcome.degraded is True
    assert "模型服务暂时不可用" in outcome.text
    assert "没有执行任何写操作" in outcome.text


async def test_booking_permission_denied_when_scope_missing() -> None:
    from app.core.security.principal import Scope

    llm = FakeLLM([ScriptedTurn(tool_calls=[ScriptedToolCall("book_trip", {})])])
    h = make_travel_harness(llm)
    await seed_planned_trip(h, make_travel_principal(), session_id="s-scope")
    # 用同一个 subject 但缺 scope 的凭据访问同一会话。
    limited = make_travel_principal(
        scopes=frozenset({Scope.CHAT, Scope.TRIP_DRAFT, Scope.KNOWLEDGE_READ})
    )
    outcome = await h.orchestrator.run(
        RunRequest(principal=limited, user_text="帮我订这个方案", session_id="s-scope")
    )
    assert outcome.finish_reason == "tool_failed"
    assert outcome.pending_action is None
    assert "无权" in outcome.text
