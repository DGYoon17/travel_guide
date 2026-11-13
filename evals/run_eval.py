"""差旅 Agent 离线评测。

两种模式：

- `--mode deterministic`：只跑金标里的确定性指标（意图/槽位/差标/审批链），不需要模型；
- `--mode live`：用真实模型跑端到端场景，记录**推理轮次、无效工具调用、token、时延**。
  简历里的数字只能来自这一模式，且必须固定数据集、模型版本与 temperature。

示例：
    python -m evals.run_eval --mode deterministic
    python -m evals.run_eval --mode live --router on
    python -m evals.run_eval --mode live --router off   # 单 ReAct 基线

报告写到 `evals/reports/`。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from app.core.agent.router import Paradigm, RouteDecision
from app.agent.travel_orchestrator import RunRequest, TravelOrchestrator
from app.core.harness.confirmation import InMemoryConfirmationStore, InMemoryIdempotencyStore
from app.core.intent.travel import TravelIntent, TravelIntentRecognizer
from app.core.memory.profile import ClaimsProfileProvider
from app.core.memory.travel_session import InMemoryTravelSessionStore, TravelSessionState
from app.core.memory.trip_draft import TripDraft
from app.core.prompt.travel_state import TravelState
from app.core.security.principal import Principal, Scope
from app.core.tools.spec import ToolContext
from app.core.tools.travel_builtin import build_travel_registry
from app.domain.procurement.models import RequesterLevel
from app.domain.travel.policy import default_travel_policy
from app.infrastructure.suppliers.mock_travel import MockTravelSupplier
from app.services.llm import LLMService
from tests.golden.travel_dataset import (
    INTENT_CASES,
    POLICY_CASES,
    REFERENCE_TODAY,
    SLOT_CASES,
    build_trip,
)

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "travel_cases.jsonl"
REPORTS = ROOT / "reports"
INVALID_CODES = {
    "invalid_arguments",
    "missing_arguments",
    "malformed_arguments_json",
    "unknown_tool",
    "precondition_not_met",
}


class CountingLLM:
    """统计模型调用次数：一轮 = 一次 stream_turn。"""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0

    @property
    def model(self) -> str:
        return self._inner.model

    async def stream_turn(self, *args: Any, **kwargs: Any) -> AsyncIterator[tuple[str, Any]]:
        self.calls += 1
        async for item in self._inner.stream_turn(*args, **kwargs):
            yield item

    async def complete(self, *args: Any, **kwargs: Any) -> str:
        return await self._inner.complete(*args, **kwargs)


class ForceReactRouter:
    """基线：所有可执行意图都走同一条 ReAct 循环，不做多范式路由。"""

    def route(self, intent: TravelIntent, confidence: float, text: str, draft: Any) -> RouteDecision:
        complexity = 0
        if intent is TravelIntent.GENERAL:
            return RouteDecision(Paradigm.DIRECT, "baseline_general", complexity, intent, confidence)
        return RouteDecision(Paradigm.REACT, "baseline_force_react", complexity, intent, confidence)


@dataclass
class CaseResult:
    case_id: str
    ok: bool
    paradigm: str
    finish_reason: str
    iterations: int
    tool_calls: int
    invalid_tool_calls: int
    tokens: int
    latency_ms: float
    detail: str = ""


@dataclass
class Summary:
    total: int = 0
    passed: int = 0
    iterations: list[int] = field(default_factory=list)
    invalid: int = 0
    tool_calls: int = 0
    tokens: list[int] = field(default_factory=list)
    latency: list[float] = field(default_factory=list)

    def add(self, result: CaseResult) -> None:
        self.total += 1
        self.passed += int(result.ok)
        self.iterations.append(result.iterations)
        self.invalid += result.invalid_tool_calls
        self.tool_calls += result.tool_calls
        self.tokens.append(result.tokens)
        self.latency.append(result.latency_ms)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": self.total,
            "success_rate": round(self.passed / self.total, 4) if self.total else 0.0,
            "mean_iterations": round(statistics.fmean(self.iterations), 3) if self.iterations else 0.0,
            "tool_calls": self.tool_calls,
            "invalid_tool_calls": self.invalid,
            "invalid_tool_call_rate": round(self.invalid / self.tool_calls, 4) if self.tool_calls else 0.0,
            "mean_tokens": round(statistics.fmean(self.tokens), 1) if self.tokens else 0.0,
            "latency_p50_ms": round(statistics.median(self.latency), 1) if self.latency else 0.0,
            "latency_p95_ms": round(_p95(self.latency), 1) if self.latency else 0.0,
        }


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return ordered[index]


def _principal() -> Principal:
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


def _load_cases(limit: int | None = None) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    return cases[:limit] if limit else cases


def _seed_state(case: dict[str, Any], principal: Principal) -> TravelSessionState:
    state = TravelSessionState(
        session_id=f"eval_{case['id']}",
        tenant_id=principal.tenant_id,
        subject_id=principal.subject_id,
        conversation_state=TravelState.PLANNING,
        draft=TripDraft(
            session_id=f"eval_{case['id']}",
            tenant_id=principal.tenant_id,
            subject_id=principal.subject_id,
        ),
    )
    from app.core.memory.slots import SlotSource

    state.draft.merge(case.get("slots") or {}, source=SlotSource.USER, turn=1)
    state.draft.confirm()
    return state


async def _preplan(orchestrator: TravelOrchestrator, state: TravelSessionState, principal: Principal) -> None:
    from app.core.tools.errors import ToolSuccess

    profile = await ClaimsProfileProvider().get(principal)
    assert profile is not None
    ctx = ToolContext(
        principal=principal,
        profile=profile,
        draft=state.draft,
        policy=default_travel_policy(version="eval-v1"),
        session_id=state.session_id,
        trace_id="eval-plan",
        today=REFERENCE_TODAY,
        supplier=orchestrator._supplier,
    )
    result = await orchestrator._registry.invoke("plan_trip", {}, ctx)
    if not isinstance(result, ToolSuccess):
        raise RuntimeError(f"preplan failed: {getattr(result, 'message', result)}")


async def run_live(limit: int | None, *, router_on: bool) -> list[CaseResult]:
    principal = _principal()
    supplier = MockTravelSupplier()
    registry = build_travel_registry()
    policy = default_travel_policy(version="eval-v1")
    results: list[CaseResult] = []

    for case in _load_cases(limit):
        llm = CountingLLM(LLMService())
        sessions = InMemoryTravelSessionStore()
        state = _seed_state(case, principal)
        await sessions.save(state)
        orchestrator = TravelOrchestrator(
            llm=llm,  # type: ignore[arg-type]
            registry=registry,
            session_store=sessions,
            confirmation_store=InMemoryConfirmationStore(),
            idempotency_store=InMemoryIdempotencyStore(),
            intent_recognizer=TravelIntentRecognizer(),
            summarizer=None,
            profile_provider=ClaimsProfileProvider(),
            policy=policy,
            supplier=supplier,
            router=None if router_on else ForceReactRouter(),  # type: ignore[arg-type]
        )
        if case.get("preplan"):
            await _preplan(orchestrator, state, principal)

        started = time.perf_counter()
        outcome = await orchestrator.run(
            RunRequest(principal=principal, user_text=case["text"], session_id=state.session_id)
        )
        elapsed = (time.perf_counter() - started) * 1000
        names = {tc.name for tc in outcome.tool_calls}
        expected = set(case.get("expected_tools") or [])
        ok = expected.issubset(names) and outcome.finish_reason == case.get("expect_finish", "stop")
        invalid = sum(1 for tc in outcome.tool_calls if tc.error_code in INVALID_CODES)
        results.append(
            CaseResult(
                case_id=case["id"],
                ok=ok,
                paradigm=outcome.paradigm,
                finish_reason=outcome.finish_reason,
                iterations=llm.calls,
                tool_calls=len(outcome.tool_calls),
                invalid_tool_calls=invalid,
                tokens=int((outcome.usage or {}).get("total_tokens") or 0),
                latency_ms=elapsed,
                detail="" if ok else f"tools={sorted(names)} finish={outcome.finish_reason}",
            )
        )
    return results


def run_deterministic() -> dict[str, Any]:
    """复用金标用例，输出可直接粘进报告的确定性指标。"""
    from app.core.intent.travel import TravelRuleEngine
    from app.core.memory.trip_draft import TripDraft
    from app.domain.travel.policy import evaluate_trip

    engine = TravelRuleEngine()
    intent_ok = sum(
        1
        for c in INTENT_CASES
        if (m := engine.classify(c.text)) and m[0] == c.expected and m[1] >= 0.85
    )
    # 相对日期用例依赖"今天"，评测时固定到金标参考日，保证可复现。
    import app.core.tools.coercion as coercion

    original_today = coercion.business_today
    coercion.business_today = lambda: REFERENCE_TODAY
    try:
        slot_ok = 0
        for case in SLOT_CASES:
            draft = TripDraft()
            from app.core.memory.slots import SlotSource

            draft.merge(case.raw, source=SlotSource.USER, turn=1)
            slot_ok += int({k: draft.value(k) for k in case.expected} == case.expected)
    finally:
        coercion.business_today = original_today

    policy = default_travel_policy(version="eval-v1")
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

    from tests.golden.travel_dataset import APPROVAL_CASES

    approval_ok = 0
    for case in APPROVAL_CASES:
        result = evaluate_trip(policy, build_trip(case), total_cny=case.total_cny, today=REFERENCE_TODAY)
        approval_ok += int(tuple(result.required_roles) == case.expected_roles)

    return {
        "intent_fast_lane": f"{intent_ok}/{len(INTENT_CASES)}",
        "slot_extraction": f"{slot_ok}/{len(SLOT_CASES)}",
        "policy_recall": f"{policy_ok}/{len(POLICY_CASES)}",
        "approval_path": f"{approval_ok}/{len(APPROVAL_CASES)}",
    }


def write_report(payload: dict[str, Any]) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS / f"eval_{stamp}.md"
    lines = [f"# 差旅 Agent 评测报告 {stamp}", "", "```json", json.dumps(payload, ensure_ascii=False, indent=2), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


async def main_async(args: argparse.Namespace) -> int:
    if args.mode == "deterministic":
        payload = {"mode": "deterministic", "metrics": run_deterministic()}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("report:", write_report(payload))
        return 0

    results = await run_live(args.limit, router_on=args.router == "on")
    summary = Summary()
    for result in results:
        summary.add(result)
    payload = {
        "mode": "live",
        "router": args.router,
        "model": LLMService().model,
        "dataset": str(DATASET.name),
        "summary": summary.as_dict(),
        "cases": [result.__dict__ for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("report:", write_report(payload))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TravelMind 离线评测")
    parser.add_argument("--mode", choices=["deterministic", "live"], default="deterministic")
    parser.add_argument("--router", choices=["on", "off"], default="on")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
