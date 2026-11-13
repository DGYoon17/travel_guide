"""滚动摘要：结构化 + 数字可追溯。

自由文本摘要在采购场景是有代价的：「预算 15000」被压成「预算约 1.5 万」，再被模型读成单价
而不是总额，就会生成错误的请购单并走错审批链。所以这里做两件事：

1. **结构化**：金额、日期、已确认事实走独立字段，可 schema 校验，不与自由文本混在一起；
2. **数字可追溯**：摘要里出现的每个金额/日期都必须能在原始对话文本或 L2 草稿状态里找到，
   否则丢弃并告警。摘要环节只允许压缩，不允许"生成"新数字。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.core.logging import get_logger
from app.core.memory.short_term import ShortTermMemory, StoredMessage
from app.infrastructure.llm.client import ChatMessage, LLMClient

logger = get_logger(__name__)

_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")
_DIGIT_RUN = re.compile(r"\d[\d,.]*")

_SUMMARY_SYSTEM = (
    "你是企业采购对话的摘要器。只做压缩，不做推断。"
    "严格输出一个 JSON 对象，不要包含其它文字或 Markdown 代码块。"
)

_SUMMARY_SCHEMA_HINT = """JSON 字段：
{
  "goals": ["用户本次要达成的目标，最多 3 条"],
  "confirmed_facts": {"字段名": "已明确的事实值"},
  "amounts_cny": {"用途": "金额数字，只写对话中原样出现过的数字"},
  "dates": {"用途": "日期，只写对话中原样出现过的日期"},
  "open_questions": ["尚未解决或待用户答复的问题"],
  "preferences": "用户的沟通与决策偏好，一到两句"
}
规则：任何金额或日期若未在对话原文出现，必须省略，不得换算、估算或补全。"""


@dataclass
class SummaryState:
    """摘要结果与它覆盖的范围。"""

    summary_text: str
    covered_messages: int
    structured: dict[str, Any] = field(default_factory=dict)
    dropped_values: list[str] = field(default_factory=list)


class MemorySummarizer:
    """token 水位超限时把较早的消息组压缩成滚动摘要。"""

    def __init__(
        self,
        llm: LLMClient,
        *,
        trigger_tokens: int | None = None,
        keep_recent_groups: int | None = None,
        summary_max_tokens: int | None = None,
    ) -> None:
        self._llm = llm
        self._trigger = trigger_tokens or settings.memory_summary_trigger_tokens
        self._keep_recent = keep_recent_groups or settings.memory_keep_recent_groups
        self._max_tokens = summary_max_tokens or settings.memory_summary_max_tokens

    def should_compress(self, memory: ShortTermMemory) -> bool:
        return memory.total_tokens() > self._trigger and memory.group_count() > self._keep_recent

    async def maybe_compress(
        self,
        memory: ShortTermMemory,
        *,
        draft: Any | None = None,
    ) -> SummaryState | None:
        """必要时压缩。返回 None 表示未触发，不会像改造前那样对空历史白调一次 LLM。"""
        if not self.should_compress(memory):
            return None

        evicted = memory.evict_oldest_groups(self._keep_recent)
        if not evicted:
            return None

        transcript = self._render_transcript(evicted)
        previous = memory.summary_text()

        try:
            structured = await self._request_structured_summary(transcript, previous)
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory.summary_failed", error=str(exc))
            # 摘要失败时保留一条最小可用的占位，避免历史"凭空消失"。
            fallback = previous or f"（已压缩 {len(evicted)} 条历史消息，摘要生成失败）"
            memory.set_summary(fallback)
            return SummaryState(summary_text=fallback, covered_messages=len(evicted))

        verified, dropped = self._verify_numbers(structured, transcript, draft)
        text = self._render_summary(verified)
        memory.set_summary(text)

        if dropped:
            logger.warning("memory.summary_values_dropped", values=dropped)

        return SummaryState(
            summary_text=text,
            covered_messages=len(evicted),
            structured=verified,
            dropped_values=dropped,
        )

    # ------------------------------------------------------------------ 内部
    def _render_transcript(self, messages: list[StoredMessage]) -> str:
        lines: list[str] = []
        for msg in messages:
            if msg.role == "tool":
                lines.append(f"[工具结果] {msg.content[:600]}")
            elif msg.tool_calls:
                calls = "，".join(f"{tc.name}({tc.arguments[:200]})" for tc in msg.tool_calls)
                lines.append(f"assistant[调用工具]: {calls}")
            else:
                lines.append(f"{msg.role}: {msg.content[:800]}")
        return "\n".join(lines)

    async def _request_structured_summary(self, transcript: str, previous: str) -> dict[str, Any]:
        prior = f"已有摘要（需与新内容合并，不要丢失其中事实）：\n{previous}\n\n" if previous else ""
        result = await self._llm.chat(
            [
                ChatMessage(role="system", content=_SUMMARY_SYSTEM),
                ChatMessage(
                    role="user",
                    content=f"{_SUMMARY_SCHEMA_HINT}\n\n{prior}待压缩对话：\n{transcript}",
                ),
            ],
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        return self._parse_json(result.content)

    def _parse_json(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = _JSON_BLOCK.search(text)
            if not match:
                raise
            data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise TypeError("summary is not a JSON object")
        return data

    def _verify_numbers(
        self,
        structured: dict[str, Any],
        transcript: str,
        draft: Any | None,
    ) -> tuple[dict[str, Any], list[str]]:
        """剔除无法在原文或草稿状态中溯源的金额与日期。"""
        transcript_digits = {m.replace(",", "") for m in _DIGIT_RUN.findall(transcript)}
        known = set(draft.known_amounts()) if draft is not None else set()
        known |= {str(v) for v in (draft.values().values() if draft is not None else [])}

        dropped: list[str] = []
        cleaned = dict(structured)

        for section in ("amounts_cny", "dates"):
            raw_section = structured.get(section)
            if not isinstance(raw_section, dict):
                cleaned[section] = {}
                continue
            kept: dict[str, Any] = {}
            for key, value in raw_section.items():
                token = str(value).strip()
                normalized = token.replace(",", "").replace("元", "").replace("￥", "")
                digits = {m.replace(",", "") for m in _DIGIT_RUN.findall(normalized)}
                traceable = (
                    token in known
                    or normalized in known
                    or normalized in transcript.replace(",", "")
                    or (digits and digits <= transcript_digits)
                )
                if traceable:
                    kept[key] = token
                else:
                    dropped.append(f"{section}.{key}={token}")
            cleaned[section] = kept

        for key in ("goals", "open_questions"):
            value = structured.get(key)
            cleaned[key] = [str(v) for v in value][:5] if isinstance(value, list) else []
        facts = structured.get("confirmed_facts")
        cleaned["confirmed_facts"] = (
            {str(k): str(v) for k, v in facts.items()} if isinstance(facts, dict) else {}
        )
        cleaned["preferences"] = str(structured.get("preferences") or "").strip()
        return cleaned, dropped

    def _render_summary(self, structured: dict[str, Any]) -> str:
        parts: list[str] = []
        if goals := structured.get("goals"):
            parts.append("用户目标：" + "；".join(goals))
        if facts := structured.get("confirmed_facts"):
            parts.append("已确认事实：" + "；".join(f"{k}={v}" for k, v in facts.items()))
        if amounts := structured.get("amounts_cny"):
            parts.append("金额（元）：" + "；".join(f"{k}={v}" for k, v in amounts.items()))
        if dates := structured.get("dates"):
            parts.append("关键日期：" + "；".join(f"{k}={v}" for k, v in dates.items()))
        if questions := structured.get("open_questions"):
            parts.append("待解决：" + "；".join(questions))
        if prefs := structured.get("preferences"):
            parts.append("用户偏好：" + prefs)
        if not parts:
            return "（历史对话无可保留要点）"
        parts.append("注：金额与审批结论以请购单草稿和审批系统为准。")
        return "\n".join(parts)
