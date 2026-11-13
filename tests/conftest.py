"""测试夹具：可编排的假 LLM 与编排器构造器。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable

from app.agent.orchestrator import ProcurementOrchestrator
from app.core.harness.confirmation import (
    InMemoryConfirmationStore,
    InMemoryIdempotencyStore,
)
from app.core.intent.recognizer import IntentRecognizer
from app.core.memory.audit import InMemoryAuditLog
from app.core.memory.profile import ClaimsProfileProvider
from app.core.memory.session_store import InMemorySessionStore
from app.core.security.principal import Principal, Scope
from app.core.tools.builtin import build_default_registry
from app.domain.procurement.models import RequesterLevel
from app.infrastructure.llm.circuit_breaker import LLMUnavailable
from app.services.llm import StreamedTurn, ToolCallAccumulator


@dataclass
class ScriptedToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = "call_1"


@dataclass
class ScriptedTurn:
    """一轮模型输出：纯文本，或一批工具调用。"""

    content: str = ""
    tool_calls: list[ScriptedToolCall] = field(default_factory=list)
    prompt_tokens: int = 100
    completion_tokens: int = 50
    finish_reason: str | None = None


class FakeLLM:
    """按脚本回放的 LLM，接口与 `LLMService` 中被编排器使用的部分一致。"""

    def __init__(
        self,
        turns: Iterable[ScriptedTurn],
        *,
        model: str = "fake-qwen",
        fail_with: str | None = None,
        complete_reply: str = "无",
    ) -> None:
        self._turns = list(turns)
        self._model = model
        self._fail_with = fail_with
        self._complete_reply = complete_reply
        self.calls: list[list[dict[str, Any]]] = []
        self.complete_calls = 0

    @property
    def model(self) -> str:
        return self._model

    @property
    def circuit_open(self) -> bool:
        return self._fail_with is not None

    async def stream_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = "auto",
        temperature: float = 0.2,
        timeout: float | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        if self._fail_with is not None:
            raise LLMUnavailable(self._fail_with)

        self.calls.append(messages)
        if not self._turns:
            raise AssertionError("FakeLLM 脚本已耗尽，说明编排器多调了一轮")
        scripted = self._turns.pop(0)

        for piece in _chunks(scripted.content):
            yield "content", piece

        turn = StreamedTurn(
            content=scripted.content,
            tool_calls=[
                ToolCallAccumulator(
                    index=i,
                    id=tc.call_id if tc.call_id != "call_1" else f"call_{i}",
                    name=tc.name,
                    arguments=json.dumps(tc.arguments, ensure_ascii=False),
                )
                for i, tc in enumerate(scripted.tool_calls)
            ],
            finish_reason=scripted.finish_reason
            or ("tool_calls" if scripted.tool_calls else "stop"),
            prompt_tokens=scripted.prompt_tokens,
            completion_tokens=scripted.completion_tokens,
        )
        yield "turn", turn

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        timeout: float | None = None,
    ) -> str:
        self.complete_calls += 1
        if self._fail_with is not None:
            raise LLMUnavailable(self._fail_with)
        return self._complete_reply


def _chunks(text: str, size: int = 8) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] if text else []


def make_principal(
    *,
    subject_id: str = "E10086",
    tenant_id: str = "acme",
    department: str = "市场部",
    scopes: frozenset[Scope] | None = None,
) -> Principal:
    return Principal(
        subject_id=subject_id,
        display_name="张三",
        tenant_id=tenant_id,
        department=department,
        requester_level=RequesterLevel.MANAGER,
        cost_center="CC-MKT-01",
        manager_id="E10001",
        budget_codes=("BUD-MKT-2026",),
        scopes=scopes
        if scopes is not None
        else frozenset(
            {Scope.CHAT, Scope.REQUISITION_DRAFT, Scope.REQUISITION_SUBMIT, Scope.KNOWLEDGE_READ}
        ),
    )


@dataclass
class Harness:
    orchestrator: ProcurementOrchestrator
    llm: FakeLLM
    sessions: InMemorySessionStore
    confirmations: InMemoryConfirmationStore
    idempotency: InMemoryIdempotencyStore
    audit: InMemoryAuditLog


def make_harness(llm: FakeLLM, *, knowledge: Any | None = None) -> Harness:
    sessions = InMemorySessionStore()
    confirmations = InMemoryConfirmationStore()
    idempotency = InMemoryIdempotencyStore()
    audit = InMemoryAuditLog()
    orchestrator = ProcurementOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        registry=build_default_registry(),
        session_store=sessions,
        confirmation_store=confirmations,
        idempotency_store=idempotency,
        audit=audit,
        intent_recognizer=IntentRecognizer(),
        summarizer=None,
        profile_provider=ClaimsProfileProvider(),
        knowledge=knowledge,
    )
    return Harness(orchestrator, llm, sessions, confirmations, idempotency, audit)


COMPLETE_REQUISITION_ARGS: dict[str, Any] = {
    "category": "software",
    "item_name": "设计协作软件",
    "quantity": 1,
    "unit_price_cny": "15000",
    "needed_by": "2026-09-15",
    "reason": "市场部设计协作",
}


# --------------------------------------------------------------------------- 差旅夹具


from types import SimpleNamespace

from app.agent.travel_orchestrator import TravelOrchestrator
from app.core.intent.travel import TravelIntentRecognizer
from app.core.memory.slots import SlotSource
from app.core.memory.travel_session import InMemoryTravelSessionStore, TravelSessionState
from app.core.memory.trip_draft import TripDraft
from app.core.prompt.travel_state import TravelState
from app.core.tools.coercion import business_today
from app.core.tools.errors import ToolSuccess
from app.core.tools.spec import ToolContext
from app.core.tools.travel_builtin import build_travel_registry
from app.domain.travel.policy import default_travel_policy
from app.infrastructure.suppliers.mock_travel import MockTravelSupplier


class FakeLongTerm:
    """进程内长期记忆，接口与 `TenantLongTermMemory` 对齐。"""

    def __init__(self, records: Iterable[str] = ()) -> None:
        self.records = list(records)
        self.written: list[str] = []

    async def recall(self, query: str, *, tenant_id: str, subject_id: str, top_k: int = 5):
        return [SimpleNamespace(text=text) for text in self.records[:top_k]]

    async def remember(self, text: str, *, tenant_id: str, subject_id: str, kind: Any = None):
        self.written.append(text)
        self.records.append(text)
        return f"ltm_{len(self.records)}"


def make_travel_principal(
    *,
    subject_id: str = "E10086",
    tenant_id: str = "acme",
    department: str = "市场部",
    level: Any = None,
    scopes: frozenset[Scope] | None = None,
) -> Principal:
    from app.domain.procurement.models import RequesterLevel

    return Principal(
        subject_id=subject_id,
        display_name="张三",
        tenant_id=tenant_id,
        department=department,
        requester_level=level or RequesterLevel.MANAGER,
        cost_center="CC-MKT-01",
        manager_id="E10001",
        budget_codes=("BUD-MKT-2026",),
        scopes=scopes
        if scopes is not None
        else frozenset(
            {
                Scope.CHAT,
                Scope.TRIP_DRAFT,
                Scope.TRIP_BOOK,
                Scope.TRIP_APPROVE,
                Scope.KNOWLEDGE_READ,
            }
        ),
    )


COMPLETE_TRIP_SLOTS: dict[str, Any] = {
    "origin_city": "上海",
    "destination_city": "北京",
    "depart_date": "2027-03-10",
    "purpose": "client",
    "trip_type": "one_way",
    "cabin_class": "economy",
    "hotel_needed": False,
    "budget_cny": "5000",
}


@dataclass
class TravelHarness:
    orchestrator: TravelOrchestrator
    llm: FakeLLM
    sessions: InMemoryTravelSessionStore
    confirmations: InMemoryConfirmationStore
    idempotency: InMemoryIdempotencyStore
    audit: InMemoryAuditLog
    supplier: MockTravelSupplier
    long_term: FakeLongTerm


def make_travel_harness(
    llm: FakeLLM,
    *,
    knowledge: Any | None = None,
    long_term: FakeLongTerm | None = None,
    policy: Any | None = None,
) -> TravelHarness:
    sessions = InMemoryTravelSessionStore()
    confirmations = InMemoryConfirmationStore()
    idempotency = InMemoryIdempotencyStore()
    audit = InMemoryAuditLog()
    supplier = MockTravelSupplier()
    ltm = long_term or FakeLongTerm()
    orchestrator = TravelOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        registry=build_travel_registry(),
        session_store=sessions,
        confirmation_store=confirmations,
        idempotency_store=idempotency,
        audit=audit,
        intent_recognizer=TravelIntentRecognizer(),
        summarizer=None,
        profile_provider=ClaimsProfileProvider(),
        policy=policy or default_travel_policy(version="test-v1"),
        knowledge=knowledge,
        supplier=supplier,
        long_term=ltm,
    )
    return TravelHarness(orchestrator, llm, sessions, confirmations, idempotency, audit, supplier, ltm)


async def seed_trip(
    sessions: InMemoryTravelSessionStore,
    principal: Principal,
    *,
    session_id: str = "s-travel",
    slots: dict[str, Any] | None = None,
) -> TravelSessionState:
    """预置一个已收集完行程要素的会话，用于验证确认门与预订链路。"""
    state = TravelSessionState(
        session_id=session_id,
        tenant_id=principal.tenant_id,
        subject_id=principal.subject_id,
        conversation_state=TravelState.PLANNING,
        draft=TripDraft(
            session_id=session_id,
            tenant_id=principal.tenant_id,
            subject_id=principal.subject_id,
        ),
    )
    state.draft.merge(slots or COMPLETE_TRIP_SLOTS, source=SlotSource.USER, turn=1)
    state.draft.confirm()
    return await sessions.save(state)


async def seed_planned_trip(
    harness: TravelHarness,
    principal: Principal,
    *,
    session_id: str = "s-planned",
) -> tuple[TravelSessionState, dict[str, Any]]:
    """预置会话 + 调用 plan_trip 生成方案（不经 LLM）。"""
    state = await seed_trip(harness.sessions, principal, session_id=session_id)
    profile = await ClaimsProfileProvider().get(principal)
    assert profile is not None
    ctx = ToolContext(
        principal=principal,
        profile=profile,
        draft=state.draft,
        policy=default_travel_policy(version="test-v1"),
        session_id=session_id,
        trace_id="t-plan",
        today=business_today(),
        supplier=harness.supplier,
    )
    result = await harness.orchestrator._registry.invoke("plan_trip", {}, ctx)
    assert isinstance(result, ToolSuccess), getattr(result, "message", result)
    state.draft = ctx.draft
    await harness.sessions.save(state)
    return state, (ctx.draft.last_plan or {})
