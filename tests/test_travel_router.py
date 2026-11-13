"""意图快慢车道与四范式路由测试。"""

from __future__ import annotations

from app.core.agent.router import AgentRouter, Paradigm
from app.core.intent.travel import (
    TravelIntent,
    TravelIntentRecognizer,
    TravelRuleEngine,
)
from app.core.memory.slots import SlotSource
from app.core.memory.trip_draft import TripDraft


def make_draft(**slots) -> TripDraft:
    draft = TripDraft()
    if slots:
        draft.merge(slots, source=SlotSource.USER, turn=1)
    return draft


def test_strong_pattern_reaches_fast_lane() -> None:
    engine = TravelRuleEngine()
    slug, confidence = engine.classify("帮我订一张去深圳的机票")
    assert slug == "booking"
    assert confidence >= 0.85


def test_weak_pattern_stays_below_threshold() -> None:
    engine = TravelRuleEngine()
    slug, confidence = engine.classify("这个出差")
    assert slug == "trip_planning"
    assert confidence < 0.85


def test_no_match_returns_none() -> None:
    assert TravelRuleEngine().classify("今天天气怎么样") is None


async def test_low_confidence_routes_to_clarify() -> None:
    recognizer = TravelIntentRecognizer()
    result = await recognizer.recognize("这个出差")
    decision = AgentRouter().route(result.intent, result.confidence, "这个出差", make_draft())
    assert result.confidence < 0.85
    assert decision.paradigm is Paradigm.CLARIFY
    assert "confidence" in decision.reason


async def test_policy_routes_direct() -> None:
    recognizer = TravelIntentRecognizer()
    result = await recognizer.recognize("公司差旅住宿标准是多少")
    decision = AgentRouter().route(result.intent, result.confidence, "公司差旅住宿标准是多少", make_draft())
    assert result.intent is TravelIntent.POLICY
    assert decision.paradigm is Paradigm.DIRECT


async def test_incomplete_slots_force_react() -> None:
    recognizer = TravelIntentRecognizer()
    text = "帮我安排下周去北京出差"
    result = await recognizer.recognize(text)
    decision = AgentRouter().route(result.intent, result.confidence, text, make_draft())
    assert decision.paradigm is Paradigm.REACT
    assert "slots_incomplete" in decision.reason


async def test_complete_trip_planning_routes_plan_execute() -> None:
    draft = make_draft(
        origin_city="上海",
        destination_city="北京",
        depart_date="2027-03-10",
        purpose="client",
    )
    recognizer = TravelIntentRecognizer()
    text = "帮我安排一下行程"
    result = await recognizer.recognize(text)
    decision = AgentRouter().route(result.intent, result.confidence, text, draft)
    assert decision.paradigm is Paradigm.PLAN_EXECUTE


async def test_booking_routes_react_even_when_complete() -> None:
    draft = make_draft(
        origin_city="上海",
        destination_city="北京",
        depart_date="2027-03-10",
        purpose="client",
    )
    recognizer = TravelIntentRecognizer()
    text = "帮我订这个方案"
    result = await recognizer.recognize(text)
    decision = AgentRouter().route(result.intent, result.confidence, text, draft)
    assert decision.paradigm is Paradigm.REACT
    assert decision.reason == "write_action_needs_gate"


def test_trip_complexity_counts_structural_signals() -> None:
    draft = make_draft(
        origin_city="上海",
        destination_city="东京",
        depart_date="2027-03-10",
        purpose="client",
        international=True,
        hotel_needed=True,
        trip_type="round_trip",
        return_date="2027-03-14",
        travelers=2,
        estimated_flight_minutes=420,
    )
    assert AgentRouter.trip_complexity(draft) == 5
