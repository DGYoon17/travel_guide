"""Agent 范式路由器。

四种范式不是营销词，而是四条真实不同的执行路径：

| 范式 | 适用 | 实现 |
|---|---|---|
| DIRECT | 差标问答、制度检索、闲聊 | RAG / 单工具 + 一次生成 |
| REACT | 需要边查边决策（比价、模糊需求、预订） | 原生 function calling 循环 |
| PLAN_EXECUTE | 可预先分解的复合行程（国际、机+酒、往返） | 模板计划 + 顺序执行 + 汇总 |
| CLARIFY | 意图置信度低 | 不调工具，先澄清 |

路由用**确定性规则**而不是让 LLM 选范式：可单测、可解释、可回归。
LLM 只在意图分类那一步参与，且低置信度时也进 CLARIFY。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.core.intent.travel import TravelIntent


class Paradigm(str, Enum):
    DIRECT = "direct"
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"
    CLARIFY = "clarify"


@dataclass(frozen=True)
class RouteDecision:
    paradigm: Paradigm
    reason: str
    complexity: int
    intent: TravelIntent
    confidence: float


class AgentRouter:
    """按意图、置信度与行程复杂度选择执行范式。"""

    def __init__(self, *, confidence_threshold: float = 0.85) -> None:
        self._threshold = confidence_threshold

    def route(
        self,
        intent: TravelIntent,
        confidence: float,
        text: str,
        draft: Any,
    ) -> RouteDecision:
        complexity = self.trip_complexity(draft)

        # 闲聊/自我介绍不需要工具。但"低置信度的 general"仍要澄清：
        # 它可能只是模型没看懂，而不是用户真的在寒暄。
        if intent is TravelIntent.GENERAL:
            if confidence >= self._threshold:
                return RouteDecision(Paradigm.DIRECT, "general_chat", complexity, intent, confidence)
            return RouteDecision(
                Paradigm.CLARIFY, "general_low_confidence", complexity, intent, confidence
            )

        # 规则和模型都没有可信意图：不猜，直接澄清。
        if intent is TravelIntent.UNKNOWN:
            return RouteDecision(
                Paradigm.CLARIFY, "intent_unknown", complexity, intent, confidence
            )

        # 低置信度不猜：先澄清。这是"低置信度触发校验"的真实落点。
        if confidence < self._threshold:
            return RouteDecision(
                Paradigm.CLARIFY,
                f"intent_confidence={confidence:.2f}<{self._threshold:.2f}",
                complexity,
                intent,
                confidence,
            )

        # 差标/制度问答走 RAG 直达，不需要多轮工具。
        if intent in (TravelIntent.POLICY, TravelIntent.RAG):
            return RouteDecision(Paradigm.DIRECT, "policy_or_rag", complexity, intent, confidence)

        # 槽位不全时先走 ReAct 抽槽：模板计划没有参数可填，硬跑只会得到确定性追问。
        try:
            missing = draft.missing_required()
        except AttributeError:
            missing = ()
        if missing and intent is not TravelIntent.GENERAL:
            return RouteDecision(
                Paradigm.REACT,
                f"slots_incomplete({len(missing)})_need_react",
                complexity,
                intent,
                confidence,
            )

        # 写操作必须走可中断的 ReAct 循环 + 确认门，不能塞进固定的计划流水线。
        if intent in (TravelIntent.BOOKING, TravelIntent.APPROVAL):
            return RouteDecision(Paradigm.REACT, "write_action_needs_gate", complexity, intent, confidence)

        if intent is TravelIntent.TRIP_PLANNING:
            return RouteDecision(
                Paradigm.PLAN_EXECUTE,
                "trip_planning_is_decomposable",
                complexity,
                intent,
                confidence,
            )

        if intent in (TravelIntent.FLIGHT_SEARCH, TravelIntent.HOTEL_SEARCH):
            if complexity >= 3:
                return RouteDecision(
                    Paradigm.PLAN_EXECUTE,
                    f"complex_trip(complexity={complexity})",
                    complexity,
                    intent,
                    confidence,
                )
            return RouteDecision(Paradigm.REACT, "single_domain_search", complexity, intent, confidence)

        return RouteDecision(Paradigm.REACT, "default", complexity, intent, confidence)

    @staticmethod
    def trip_complexity(draft: Any) -> int:
        """行程复杂度 0~5。只看结构化槽位，不解析自然语言。"""
        def value(name: str) -> Any:
            try:
                return draft.value(name)
            except AttributeError:
                return None

        score = 0
        if value("international"):
            score += 1
        if value("hotel_needed"):
            score += 1
        if value("return_date") or value("trip_type") == "round_trip":
            score += 1
        try:
            if int(value("travelers") or 1) > 1:
                score += 1
        except (TypeError, ValueError):
            pass
        try:
            if int(value("estimated_flight_minutes") or 0) >= 360:
                score += 1
        except (TypeError, ValueError):
            pass
        return score

    @staticmethod
    def clarification(draft: Any) -> str:
        """低置信度时的确定性澄清话术。缺槽位优先追问，否则让用户在三个动作里选。"""
        try:
            missing = draft.missing_required()
        except AttributeError:
            missing = ()
        if missing:
            return draft.missing_prompt()
        return (
            "我不太确定你想做哪件事。你是想：\n"
            "1) 查询差旅标准（舱位/住宿/提前天数）\n"
            "2) 规划一次出差行程\n"
            "3) 直接预订机票或酒店\n"
            "可以补充一句目的地和日期，我就按对应流程处理。"
        )


class BaselineRouter:
    """A/B 基线：所有可执行意图都走同一条 ReAct 循环，不做多范式路由。

    只用于压测/离线对比，不参与默认装配。
    """

    def route(
        self,
        intent: TravelIntent,
        confidence: float,
        text: str,
        draft: Any,
    ) -> RouteDecision:
        complexity = 0
        if intent is TravelIntent.GENERAL:
            return RouteDecision(Paradigm.DIRECT, "baseline_general", complexity, intent, confidence)
        return RouteDecision(Paradigm.REACT, "baseline_force_react", complexity, intent, confidence)


__all__ = ["AgentRouter", "BaselineRouter", "Paradigm", "RouteDecision"]
