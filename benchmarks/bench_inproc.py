"""编排层压测（不依赖网络与真实模型）。

思路：把 LLM 抽象成一个**固定时延**的可编程对象，只压服务端真正负责的部分：
路由、意图、工具执行、记忆裁剪、确认门、会话持久化。

这样可以回答：
- 并发上升时，编排层自身是否稳定、是否线性扩展；
- 多范式路由相对"全部走 ReAct"少调用几次模型；
- 在固定模型时延下，端到端 P50/P95 差多少。

**它不能回答**：真实模型的 token 生成速度、限流、上下文长度对时延的影响。
那些必须用 `--live` 的小并发实验，见 benchmarks/README.md。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, AsyncIterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.travel_orchestrator import RunRequest, TravelOrchestrator
from app.core.agent.router import AgentRouter, Paradigm, RouteDecision
from app.core.harness.confirmation import InMemoryConfirmationStore, InMemoryIdempotencyStore
from app.core.intent.travel import TravelIntent, TravelIntentRecognizer
from app.core.memory.profile import ClaimsProfileProvider
from app.core.memory.slots import SlotSource
from app.core.memory.travel_session import InMemoryTravelSessionStore, TravelSessionState
from app.core.memory.trip_draft import TripDraft
from app.core.prompt.travel_state import TravelState
from app.core.security.principal import Principal, Scope
from app.core.tools.spec import ToolContext
from app.core.tools.travel_builtin import build_travel_registry
from app.domain.procurement.models import RequesterLevel
from app.domain.travel.policy import default_travel_policy
from app.infrastructure.suppliers.mock_travel import MockTravelSupplier
from app.services.llm import StreamedTurn, ToolCallAccumulator

REPORTS = ROOT / "benchmarks" / "reports"
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


class ForceReactRouter:
    """基线：所有可执行意图都走同一条 ReAct 循环。"""

    def route(self, intent: TravelIntent, confidence: float, text: str, draft: Any) -> RouteDecision:
        if intent is TravelIntent.GENERAL:
            return RouteDecision(Paradigm.DIRECT, "baseline_general", 0, intent, confidence)
        return RouteDecision(Paradigm.REACT, "baseline_force_react", 0, intent, confidence)


class LatencyLLM:
    """固定时延的可编程模型：按工具集与消息状态决定是否先发一个工具调用。"""

    model = "bench-mock"

    def __init__(self, delay_ms: float, *, chunk_delay_ms: float = 0.0, chunks: int = 5) -> None:
        self._delay = delay_ms / 1000.0
        self._chunk_delay = chunk_delay_ms / 1000.0
        self._chunks = chunks
        self.calls = 0

    async def stream_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = "auto",
        temperature: float = 0.2,
        timeout: float | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        self.calls += 1
        await asyncio.sleep(self._delay)

        tool_names = {
            item.get("function", {}).get("name", "")
            for item in (tools or [])
            if isinstance(item, dict)
        }
        user_text = ""
        has_tool_result = False
        for message in messages:
            if message.get("role") == "user":
                user_text = str(message.get("content") or "")
            if message.get("role") == "tool":
                has_tool_result = True

        # 基线 ReAct：模型必须先调一个工具，才有信息回答。
        if tools and not has_tool_result:
            tool_name = None
            if "get_travel_policy" in tool_names and "标准" in user_text:
                tool_name = "get_travel_policy"
            elif "plan_trip" in tool_names and "安排" in user_text:
                tool_name = "plan_trip"
            elif "book_trip" in tool_names and "订" in user_text:
                tool_name = "book_trip"
            if tool_name:
                turn = StreamedTurn(
                    content="",
                    tool_calls=[
                        ToolCallAccumulator(
                            index=0,
                            id="call_bench",
                            name=tool_name,
                            arguments="{}",
                        )
                    ],
                    finish_reason="tool_calls",
                    prompt_tokens=100,
                    completion_tokens=20,
                )
                yield "turn", turn
                return

        answer = "已按差标与行程给出结果。"
        pieces = [answer[i : i + max(1, len(answer) // self._chunks)] for i in range(0, len(answer), max(1, len(answer) // self._chunks))]
        for piece in pieces[: self._chunks]:
            yield "content", piece
            await asyncio.sleep(self._chunk_delay)
        yield "turn", StreamedTurn(
            content=answer,
            finish_reason="stop",
            prompt_tokens=120,
            completion_tokens=30,
        )

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
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


async def seed_state(
    sessions: InMemoryTravelSessionStore,
    *,
    session_id: str,
    plan: bool = False,
    supplier: MockTravelSupplier | None = None,
) -> TravelSessionState:
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
    if plan:
        supplier = supplier or MockTravelSupplier()
        context = ToolContext(
            principal=p,
            profile=await ClaimsProfileProvider().get(p),  # type: ignore[arg-type]
            draft=state.draft,
            policy=default_travel_policy(version="bench-v1"),
            session_id=session_id,
            trace_id="bench-plan",
            today=datetime(2026, 11, 11).date(),
            supplier=supplier,
        )
        result = await build_travel_registry().invoke("plan_trip", {}, context)
        if not getattr(result, "data", None):
            raise RuntimeError(f"plan failed: {getattr(result, 'message', result)}")
        state.draft = context.draft
    return await sessions.save(state)


def build_orchestrator(
    llm: LatencyLLM,
    sessions: InMemoryTravelSessionStore,
    *,
    router_on: bool,
) -> TravelOrchestrator:
    return TravelOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        registry=build_travel_registry(),
        session_store=sessions,
        confirmation_store=InMemoryConfirmationStore(),
        idempotency_store=InMemoryIdempotencyStore(),
        intent_recognizer=TravelIntentRecognizer(),
        summarizer=None,
        profile_provider=ClaimsProfileProvider(),
        policy=default_travel_policy(version="bench-v1"),
        supplier=MockTravelSupplier(),
        router=AgentRouter() if router_on else ForceReactRouter(),  # type: ignore[arg-type]
    )


@dataclass
class Sample:
    latency_ms: float
    llm_calls: int
    tool_calls: int
    ok: bool


@dataclass
class Scenario:
    name: str
    text: str
    plan_seed: bool = False


SCENARIOS = (
    Scenario("policy", "公司差旅住宿标准是多少"),
    Scenario("plan", "帮我安排一下行程"),
    Scenario("chat", "你好"),
    Scenario("booking_gate", "帮我订这个方案", plan_seed=True),
)


async def worker(
    scenario: Scenario,
    *,
    router_on: bool,
    requests: int,
    delay_ms: float,
    worker_id: int,
) -> list[Sample]:
    sessions = InMemoryTravelSessionStore()
    llm = LatencyLLM(delay_ms)
    orchestrator = build_orchestrator(llm, sessions, router_on=router_on)
    p = principal()
    samples: list[Sample] = []
    for index in range(requests):
        session_id = f"{scenario.name}-{router_on}-{worker_id}-{index}"
        await seed_state(
            sessions,
            session_id=session_id,
            plan=scenario.plan_seed,
            supplier=orchestrator._supplier,
        )
        before = llm.calls
        started = time.perf_counter()
        try:
            outcome = await orchestrator.run(
                RunRequest(principal=p, user_text=scenario.text, session_id=session_id)
            )
            elapsed = (time.perf_counter() - started) * 1000
            samples.append(
                Sample(
                    latency_ms=elapsed,
                    llm_calls=llm.calls - before,
                    tool_calls=len(outcome.tool_calls),
                    ok=outcome.finish_reason in {"stop", "confirmation_required", "clarification_required"},
                )
            )
        except Exception:  # noqa: BLE001
            elapsed = (time.perf_counter() - started) * 1000
            samples.append(Sample(latency_ms=elapsed, llm_calls=llm.calls - before, tool_calls=0, ok=False))
    return samples


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


async def run_case(scenario: Scenario, *, router_on: bool, concurrency: int, requests: int, delay_ms: float) -> dict[str, Any]:
    started = time.perf_counter()
    batches = await asyncio.gather(
        *(
            worker(scenario, router_on=router_on, requests=requests, delay_ms=delay_ms, worker_id=i)
            for i in range(concurrency)
        )
    )
    wall = time.perf_counter() - started
    samples = [sample for batch in batches for sample in batch]
    latencies = [sample.latency_ms for sample in samples]
    return {
        "scenario": scenario.name,
        "mode": "router_on" if router_on else "force_react",
        "concurrency": concurrency,
        "requests": len(samples),
        "rps": round(len(samples) / wall, 2) if wall else 0.0,
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(p95(latencies), 1),
        "mean_ms": round(statistics.fmean(latencies), 1),
        "llm_calls_per_req": round(statistics.fmean(s.llm_calls for s in samples), 2),
        "tool_calls_per_req": round(statistics.fmean(s.tool_calls for s in samples), 2),
        "errors": sum(0 if s.ok else 1 for s in samples),
    }


async def main_async(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for router_on in (True, False):
            for concurrency in args.concurrency:
                row = await run_case(
                    scenario,
                    router_on=router_on,
                    concurrency=concurrency,
                    requests=args.requests,
                    delay_ms=args.delay_ms,
                )
                rows.append(row)
                print(
                    f"{row['scenario']:12s} {row['mode']:12s} c={concurrency:>3d} "
                    f"p50={row['p50_ms']:7.1f}ms p95={row['p95_ms']:7.1f}ms "
                    f"llm/req={row['llm_calls_per_req']:.2f} err={row['errors']}"
                )
    payload = {
        "mode": "inproc_mock_llm",
        "delay_ms": args.delay_ms,
        "requests_per_worker": args.requests,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"inproc_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report:", path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="编排层并发压测（假模型固定时延）")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 10, 30])
    parser.add_argument("--requests", type=int, default=10, help="每个并发 worker 的请求数")
    parser.add_argument("--delay-ms", type=float, default=30.0, help="每次 LLM 调用的固定时延")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
