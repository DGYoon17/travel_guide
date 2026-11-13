"""滚动摘要测试：结构化 + 数字可追溯（摘要只许压缩，不许生成新数字）。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.core.memory.draft import RequisitionDraft, SlotSource
from app.core.memory.short_term import ShortTermMemory, StoredMessage
from app.core.memory.summary import MemorySummarizer


@dataclass
class _Result:
    content: str
    model: str = "fake"
    prompt_tokens: int = 10
    completion_tokens: int = 20
    raw: dict = None  # type: ignore[assignment]


class FakeSummaryLLM:
    def __init__(self, payload: dict | str) -> None:
        self._payload = payload
        self.calls = 0

    async def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        body = self._payload if isinstance(self._payload, str) else json.dumps(
            self._payload, ensure_ascii=False
        )
        return _Result(content=body, raw={})


def _memory_with_history(groups: int = 20) -> ShortTermMemory:
    memory = ShortTermMemory(max_tokens=10**6, max_groups=10**6)
    for i in range(groups):
        memory.append(StoredMessage(role="user", content=f"第 {i} 轮：预算 15000 元，需要设计协作软件"))
        memory.append(StoredMessage(role="assistant", content=f"第 {i} 轮回复：已了解"))
    return memory


async def test_no_compression_below_threshold_avoids_wasted_llm_call() -> None:
    """改造前 threshold 小于 window，会拿空 transcript 白调一次 LLM。"""
    llm = FakeSummaryLLM({"goals": []})
    summarizer = MemorySummarizer(llm, trigger_tokens=10**9, keep_recent_groups=4)  # type: ignore[arg-type]
    memory = _memory_with_history(5)

    assert await summarizer.maybe_compress(memory) is None
    assert llm.calls == 0, "未触发压缩时不应调用 LLM"


async def test_compression_writes_pinned_summary_and_keeps_recent() -> None:
    llm = FakeSummaryLLM(
        {
            "goals": ["采购设计协作软件"],
            "confirmed_facts": {"品类": "software"},
            "amounts_cny": {"预算": "15000"},
            "dates": {},
            "open_questions": ["需求日期未定"],
            "preferences": "偏好既有供应商",
        }
    )
    summarizer = MemorySummarizer(llm, trigger_tokens=1, keep_recent_groups=4)  # type: ignore[arg-type]
    memory = _memory_with_history(20)

    state = await summarizer.maybe_compress(memory)
    assert state is not None
    assert state.covered_messages > 0

    messages = memory.as_messages()
    assert messages[0]["role"] == "system"
    assert "[历史摘要]" in messages[0]["content"]
    assert "15000" in messages[0]["content"]
    assert memory.group_count() == 4


async def test_untraceable_amounts_are_dropped() -> None:
    """摘要报了一个原文和草稿里都没有的金额 → 必须丢弃，不能沉淀成"事实"。"""
    llm = FakeSummaryLLM(
        {
            "goals": ["采购软件"],
            "confirmed_facts": {},
            # 15000 在原文出现过；88888 是凭空生成的。
            "amounts_cny": {"预算": "15000", "总额": "88888"},
            "dates": {},
            "open_questions": [],
            "preferences": "",
        }
    )
    summarizer = MemorySummarizer(llm, trigger_tokens=1, keep_recent_groups=2)  # type: ignore[arg-type]
    memory = _memory_with_history(10)

    draft = RequisitionDraft()
    draft.merge({"unit_price_cny": "15000"}, source=SlotSource.USER, turn=1)

    state = await summarizer.maybe_compress(memory, draft=draft)
    assert state is not None
    assert state.structured["amounts_cny"] == {"预算": "15000"}
    assert any("88888" in d for d in state.dropped_values)
    assert "88888" not in state.summary_text


async def test_draft_state_makes_derived_totals_traceable() -> None:
    """草稿里算出来的总额虽然不在对话原文里，但可溯源到 L2，应保留。"""
    llm = FakeSummaryLLM(
        {
            "goals": [],
            "confirmed_facts": {},
            "amounts_cny": {"预估总额": "45000.00"},
            "dates": {},
            "open_questions": [],
            "preferences": "",
        }
    )
    summarizer = MemorySummarizer(llm, trigger_tokens=1, keep_recent_groups=2)  # type: ignore[arg-type]
    memory = _memory_with_history(8)

    draft = RequisitionDraft()
    draft.last_requisition = {"estimated_total_cny": "45000.00"}

    state = await summarizer.maybe_compress(memory, draft=draft)
    assert state is not None
    assert state.structured["amounts_cny"] == {"预估总额": "45000.00"}
    assert state.dropped_values == []


async def test_malformed_summary_falls_back_without_losing_history_marker() -> None:
    llm = FakeSummaryLLM("这不是 JSON")
    summarizer = MemorySummarizer(llm, trigger_tokens=1, keep_recent_groups=2)  # type: ignore[arg-type]
    memory = _memory_with_history(8)

    state = await summarizer.maybe_compress(memory)
    assert state is not None
    assert "摘要生成失败" in state.summary_text
    # 历史被压缩掉了，但至少留下明确标记，不会让上下文"凭空消失"。
    assert "[历史摘要]" in memory.as_messages()[0]["content"]


async def test_summary_always_notes_authoritative_source() -> None:
    llm = FakeSummaryLLM(
        {
            "goals": ["采购软件"],
            "confirmed_facts": {},
            "amounts_cny": {},
            "dates": {},
            "open_questions": [],
            "preferences": "",
        }
    )
    summarizer = MemorySummarizer(llm, trigger_tokens=1, keep_recent_groups=2)  # type: ignore[arg-type]
    state = await summarizer.maybe_compress(_memory_with_history(8))
    assert state is not None
    assert "以请购单草稿和审批系统为准" in state.summary_text
