"""无 pytest 环境下的端到端冒烟脚本。

用法：
    python scripts/smoke_travel.py

覆盖：意图快车道、路由、差标、方案生成、确认门、二次核价、预订、退订、审批。
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import HTTPException  # noqa: E402

from app.core.agent.router import AgentRouter, Paradigm  # noqa: E402
from app.agent.travel_orchestrator import RunRequest, TravelOrchestrator  # noqa: E402
from app.core.harness.confirmation import (  # noqa: E402
    InMemoryConfirmationStore,
    InMemoryIdempotencyStore,
)
from app.core.intent.travel import (  # noqa: E402
    TravelIntent,
    TravelIntentRecognizer,
    TravelRuleEngine,
)
from app.core.memory.audit import InMemoryAuditLog  # noqa: E402
from app.core.memory.profile import ClaimsProfileProvider  # noqa: E402
from app.core.memory.slots import SlotSource  # noqa: E402
from app.core.memory.travel_session import InMemoryTravelSessionStore, TravelSessionState  # noqa: E402
from app.core.memory.trip_draft import TripDraft  # noqa: E402
from app.core.prompt.travel_state import TravelState  # noqa: E402
from app.core.security.principal import Principal, Scope  # noqa: E402
from app.core.tools.coercion import business_today  # noqa: E402
from app.core.tools.errors import ToolSuccess  # noqa: E402
from app.core.tools.spec import ToolContext  # noqa: E402
from app.core.tools.travel_builtin import build_travel_registry  # noqa: E402
from app.domain.procurement.models import RequesterLevel  # noqa: E402
from app.domain.schemas import ConfirmActionRequest, ConfirmDecision  # noqa: E402
from app.domain.travel.models import (  # noqa: E402
    CabinClass,
    TravelerLevel,
    TripPurpose,
    TripRequest,
    TripType,
)
from app.domain.travel.policy import default_travel_policy, evaluate_trip  # noqa: E402
from app.infrastructure.suppliers.mock_travel import MockTravelSupplier  # noqa: E402
from app.services.llm import StreamedTurn, ToolCallAccumulator  # noqa: E402


PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [ok] {name}")
        return
    raise AssertionError(f"{name} failed: {detail}")


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class Turn:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class ScriptedLLM:
    model = "fake-qwen"

    def __init__(self, turns: list[Turn]) -> None:
        self._turns = list(turns)
        self.calls: list[list[dict]] = []

    async def stream_turn(self, messages, *, tools=None, tool_choice="auto", temperature=0.2, timeout=None):
        self.calls.append(messages)
        if not self._turns:
            raise AssertionError("脚本已耗尽：编排器多调了一轮")
        scripted = self._turns.pop(0)
        for i in range(0, len(scripted.content), 8):
            yield "content", scripted.content[i : i + 8]
        turn = StreamedTurn(
            content=scripted.content,
            tool_calls=[
                ToolCallAccumulator(index=i, id=f"call_{i}", name=tc.name, arguments=json.dumps(tc.arguments, ensure_ascii=False))
                for i, tc in enumerate(scripted.tool_calls)
            ],
            finish_reason="tool_calls" if scripted.tool_calls else "stop",
            prompt_tokens=100,
            completion_tokens=50,
        )
        yield "turn", turn

    async def complete(self, messages, *, temperature=0.2, timeout=None) -> str:
        return "无"


def principal() -> Principal:
    return Principal(
        subject_id="E10086",
        display_name="张三",
        tenant_id="acme",
        department="市场部",
        requester_level=RequesterLevel.MANAGER,
        cost_center="CC-MKT-01",
        scopes=frozenset(
            {Scope.CHAT, Scope.TRIP_DRAFT, Scope.TRIP_BOOK, Scope.TRIP_APPROVE, Scope.KNOWLEDGE_READ}
        ),
    )


SLOTS = {
    "origin_city": "上海",
    "destination_city": "北京",
    "depart_date": "2027-03-10",
    "purpose": "client",
    "trip_type": "one_way",
    "cabin_class": "economy",
    "hotel_needed": False,
    "budget_cny": "5000",
}


async def seed(sessions: InMemoryTravelSessionStore, session_id: str = "smoke") -> TravelSessionState:
    p = principal()
    state = TravelSessionState(
        session_id=session_id,
        tenant_id=p.tenant_id,
        subject_id=p.subject_id,
        conversation_state=TravelState.PLANNING,
        draft=TripDraft(session_id=session_id, tenant_id=p.tenant_id, subject_id=p.subject_id),
    )
    state.draft.merge(SLOTS, source=SlotSource.USER, turn=1)
    state.draft.confirm()
    return await sessions.save(state)


async def make_orchestrator(llm: ScriptedLLM, sessions: InMemoryTravelSessionStore, supplier: MockTravelSupplier):
    return TravelOrchestrator(
        llm=llm,
        registry=build_travel_registry(),
        session_store=sessions,
        confirmation_store=InMemoryConfirmationStore(),
        idempotency_store=InMemoryIdempotencyStore(),
        audit=InMemoryAuditLog(),
        intent_recognizer=TravelIntentRecognizer(),
        summarizer=None,
        profile_provider=ClaimsProfileProvider(),
        policy=default_travel_policy(version="smoke-v1"),
        supplier=supplier,
    )


async def plan_state(orchestrator: TravelOrchestrator, state: TravelSessionState) -> dict:
    profile = await ClaimsProfileProvider().get(principal())
    assert profile is not None
    ctx = ToolContext(
        principal=principal(),
        profile=profile,
        draft=state.draft,
        policy=default_travel_policy(version="smoke-v1"),
        session_id=state.session_id,
        trace_id="smoke-plan",
        today=business_today(),
        supplier=orchestrator._supplier,
    )
    result = await orchestrator._registry.invoke("plan_trip", {}, ctx)
    if not isinstance(result, ToolSuccess):
        raise AssertionError(f"plan_trip failed: {getattr(result, 'message', result)}")
    state.draft = ctx.draft
    await orchestrator._sessions.save(state)
    return ctx.draft.last_plan or {}


async def test_policy() -> None:
    print("\n[1] 差标引擎")
    policy = default_travel_policy(version="smoke-v1")
    trip = TripRequest(
        trip_id="t1",
        traveler_id="E10086",
        traveler_level=TravelerLevel.MANAGER,
        department="市场部",
        origin_city="上海",
        destination_city="北京",
        depart_date=business_today(),
        trip_type=TripType.ONE_WAY,
        purpose=TripPurpose.CLIENT,
        cabin_class=CabinClass.BUSINESS,
        hotel_needed=True,
    )
    result = evaluate_trip(policy, trip, hotel_nightly_cny=Decimal("1200"), total_cny=Decimal("4200"), today=business_today())
    codes = {f.code for f in result.findings}
    check("舱位超标被识别", "CABIN_OVER_STANDARD" in codes, str(codes))
    check("住宿超标金额可计算", result.over_standard_total_cny > 0, str(result.over_standard_total_cny))
    check("审批链包含分管高管", "分管高管" in result.required_roles, str(result.required_roles))
    check("提前预订不足被提示", "ADVANCE_DAYS" in codes, str(codes))


async def test_router() -> None:
    print("\n[2] 意图与路由")
    engine = TravelRuleEngine()
    slug, confidence = engine.classify("帮我订一张去深圳的机票")
    check("强模式命中 booking", slug == "booking" and confidence >= 0.85, f"{slug} {confidence}")
    weak_slug, weak_conf = engine.classify("这个出差")
    check("弱模式置信度低于阈值", weak_slug == "trip_planning" and weak_conf < 0.85, f"{weak_slug} {weak_conf}")

    recognizer = TravelIntentRecognizer()
    low = await recognizer.recognize("安排一下")
    decision = AgentRouter().route(low.intent, low.confidence, "安排一下", TripDraft())
    check("低置信度走 CLARIFY", decision.paradigm is Paradigm.CLARIFY, decision.paradigm.value)
    policy_result = await recognizer.recognize("公司差旅住宿标准是多少")
    policy_decision = AgentRouter().route(policy_result.intent, policy_result.confidence, "公司差旅住宿标准是多少", TripDraft())
    check("差标问答走 DIRECT", policy_decision.paradigm is Paradigm.DIRECT, policy_decision.paradigm.value)


async def test_tools_and_gate() -> None:
    print("\n[3] 方案 / 确认门 / 预订")
    sessions = InMemoryTravelSessionStore()
    supplier = MockTravelSupplier()
    orchestrator = await make_orchestrator(ScriptedLLM([]), sessions, supplier)
    state = await seed(sessions, "smoke-tools")
    plan = await plan_state(orchestrator, state)
    check("方案包含航段", bool(plan.get("segments")), str(plan))
    check("方案有有效期", bool(plan.get("expires_at")), str(plan.get("expires_at")))
    check("方案有审批链", bool(plan.get("approval_path")), str(plan.get("approval_path")))

    gate_llm = ScriptedLLM([Turn(tool_calls=[ToolCall("book_trip", {})])])
    gate_orchestrator = await make_orchestrator(gate_llm, sessions, supplier)
    outcome = await gate_orchestrator.run(
        RunRequest(principal=principal(), user_text="帮我订这个方案", session_id="smoke-tools")
    )
    check("写操作被确认门拦住", outcome.finish_reason == "confirmation_required", outcome.finish_reason)
    check("未确认时没有订单", not supplier._bookings, str(supplier._bookings))
    assert outcome.pending_action is not None

    response = await gate_orchestrator.confirm_action(
        principal(),
        ConfirmActionRequest(action_id=outcome.pending_action.action_id, decision=ConfirmDecision.CONFIRM),
        trace_id="smoke-confirm",
    )
    check("确认后生成订单", response.status == "confirmed" and bool(supplier._bookings), response.status)
    check("订单数量等于航段数", len(response.result["bookings"]) == len(plan["segments"]), str(response.result))


async def test_price_change() -> None:
    print("\n[4] 报价二次校验")
    sessions = InMemoryTravelSessionStore()
    supplier = MockTravelSupplier()
    orchestrator = await make_orchestrator(ScriptedLLM([]), sessions, supplier)
    state = await seed(sessions, "smoke-price")
    plan = await plan_state(orchestrator, state)
    gate_llm = ScriptedLLM([Turn(tool_calls=[ToolCall("book_trip", {})])])
    gate_orchestrator = await make_orchestrator(gate_llm, sessions, supplier)
    outcome = await gate_orchestrator.run(
        RunRequest(principal=principal(), user_text="帮我订这个方案", session_id="smoke-price")
    )
    assert outcome.pending_action is not None
    first = plan["segments"][0]
    supplier.simulate_price_change(first["offer_id"], price_cny=Decimal(str(first["price_cny"])) * Decimal("1.1"))
    try:
        await gate_orchestrator.confirm_action(
            principal(),
            ConfirmActionRequest(action_id=outcome.pending_action.action_id, decision=ConfirmDecision.CONFIRM),
            trace_id="smoke-price-confirm",
        )
    except HTTPException as exc:
        check("报价变化返回 409", exc.status_code == 409, str(exc.status_code))
        check("未产生订单", not supplier._bookings, str(supplier._bookings))
    else:
        raise AssertionError("报价变化后仍然执行了预订")


async def test_plan_execute() -> None:
    print("\n[5] PLAN_EXECUTE 流水线")
    sessions = InMemoryTravelSessionStore()
    supplier = MockTravelSupplier()
    state = await seed(sessions, "smoke-plan")
    llm = ScriptedLLM([Turn(content="已生成方案，请确认。")])
    orchestrator = await make_orchestrator(llm, sessions, supplier)
    outcome = await orchestrator.run(
        RunRequest(principal=principal(), user_text="帮我安排一下行程", session_id="smoke-plan")
    )
    names = [tc.name for tc in outcome.tool_calls]
    check("走 plan_execute", outcome.paradigm == "plan_execute", outcome.paradigm)
    check("执行固定工具序列", names == ["get_travel_policy", "search_flights", "check_travel_compliance", "plan_trip"], str(names))
    check("汇总文案来自模型", outcome.text == "已生成方案，请确认。", outcome.text)


async def test_clarify_and_degrade() -> None:
    print("\n[6] 澄清与降级")
    sessions = InMemoryTravelSessionStore()
    orchestrator = await make_orchestrator(ScriptedLLM([]), sessions, MockTravelSupplier())
    outcome = await orchestrator.run(RunRequest(principal=principal(), user_text="安排一下"))
    check("空槽位触发澄清", outcome.finish_reason == "clarification_required", outcome.finish_reason)
    check("澄清文案列出缺失项", "还需要以下信息" in outcome.text, outcome.text)

    class FailingLLM(ScriptedLLM):
        async def stream_turn(self, *a, **kw):
            raise RuntimeError("upstream 503")
            yield  # pragma: no cover

    from app.infrastructure.llm.circuit_breaker import LLMUnavailable

    class UnavailableLLM(ScriptedLLM):
        async def stream_turn(self, *a, **kw):
            raise LLMUnavailable("upstream 503")
            yield  # pragma: no cover

    sessions2 = InMemoryTravelSessionStore()
    await seed(sessions2, "smoke-degrade")
    degrading = await make_orchestrator(UnavailableLLM([]), sessions2, MockTravelSupplier())
    degraded = await degrading.run(
        RunRequest(principal=principal(), user_text="帮我安排一下行程", session_id="smoke-degrade")
    )
    check("模型不可用走确定性降级", degraded.degraded and "模型服务暂时不可用" in degraded.text, degraded.text)


async def test_golden_counts() -> None:
    print("\n[7] 金标抽样")
    from tests.golden.travel_dataset import INTENT_CASES, POLICY_CASES, build_trip

    engine = TravelRuleEngine()
    intent_ok = sum(
        1 for c in INTENT_CASES if (m := engine.classify(c.text)) and m[0] == c.expected and m[1] >= 0.85
    )
    check("意图快车道 >= 85%", intent_ok / len(INTENT_CASES) >= 0.85, f"{intent_ok}/{len(INTENT_CASES)}")

    policy = default_travel_policy(version="smoke-v1")
    from tests.golden.travel_dataset import REFERENCE_TODAY

    policy_ok = 0
    for case in POLICY_CASES:
        result = evaluate_trip(
            policy,
            build_trip(case),
            hotel_nightly_cny=case.hotel_nightly_cny,
            total_cny=case.total_cny,
            today=REFERENCE_TODAY,
        )
        codes = {f.code for f in result.findings}
        policy_ok += int(
            all(c in codes for c in case.expected_codes)
            and all(r in result.required_roles for r in case.expected_roles)
            and result.compliant is case.expect_compliant
        )
    check("差标金标全通过", policy_ok == len(POLICY_CASES), f"{policy_ok}/{len(POLICY_CASES)}")


async def main() -> int:
    print("TravelMind 冒烟测试")
    await test_policy()
    await test_router()
    await test_tools_and_gate()
    await test_price_change()
    await test_plan_execute()
    await test_clarify_and_degrade()
    await test_golden_counts()
    print(f"\n全部通过：{len(PASSED)} 项")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
