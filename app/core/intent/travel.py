"""差旅意图识别：规则快车道 + LLM 慢车道。

与采购版的区别不只是词表，而是**置信度分层**：
- 命中强模式（「订机票」「查差标」）→ 0.86 起，直接走快车道；
- 只命中弱模式（「安排一下」「看看这个」）→ 0.68~0.80，进入慢车道或澄清；
- 低于阈值时不猜，交给 `AgentRouter` 触发 CLARIFY。

这正是简历里"低置信度触发校验"应有的实现方式：阈值两侧都必须可达，而不是写一个
永远不成立的 `< 0.82`。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from app.core.intent.llm_classifier import LLMClassification, LLMIntentClassifier


class TravelIntent(str, Enum):
    FLIGHT_SEARCH = "flight_search"
    HOTEL_SEARCH = "hotel_search"
    TRIP_PLANNING = "trip_planning"
    BOOKING = "booking"
    POLICY = "policy"
    APPROVAL = "approval"
    TRIP_STATUS = "trip_status"
    RAG = "rag"
    GENERAL = "general"
    #: 规则与模型都没给出可信意图时的显式状态。不要用 GENERAL 兜底：
    #: "安排一下" 不是闲聊，它需要澄清，而不是直接走无工具回答。
    UNKNOWN = "unknown"


BusinessIntent = TravelIntent

_SLUG_TO_ENUM: dict[str, TravelIntent] = {i.value: i for i in TravelIntent}
_DEFAULT_ALLOWED: tuple[str, ...] = tuple(i.value for i in TravelIntent)

_TIE_BREAK_PRIORITY: dict[str, int] = {
    "booking": 100,
    "approval": 96,
    "trip_planning": 92,
    "flight_search": 84,
    "hotel_search": 82,
    "policy": 80,
    "trip_status": 70,
    "rag": 60,
}

TRAVEL_CLASSIFIER_SYSTEM = (
    "你是企业差旅助手的意图分类器。根据用户输入，从给定意图中选一个最匹配的。"
    "必须只输出一个 JSON 对象，字段：intent_slug (string), confidence (0-1 小数), "
    "rationale (简短中文理由)。不要输出其它文字。"
)


@dataclass(frozen=True)
class TravelRule:
    intent_slug: str
    strong: tuple[re.Pattern[str], ...] = ()
    weak: tuple[re.Pattern[str], ...] = ()


class TravelRuleEngine:
    """关键词/正则规则。强模式代表明确意图，弱模式只代表"可能相关"。"""

    def __init__(self) -> None:
        self._rules: list[TravelRule] = [
            TravelRule(
                "flight_search",
                strong=(
                    re.compile(r"(查|搜|看|找).{0,12}(机票|航班|飞机票)"),
                    re.compile(r"(机票|航班).{0,6}(价格|报价|班次|直飞)"),
                ),
                weak=(
                    re.compile(r"(机票|航班|飞机)"),
                    re.compile(r"(飞去|飞到)"),
                ),
            ),
            TravelRule(
                "hotel_search",
                strong=(
                    re.compile(r"(查|搜|找|订).{0,6}(酒店|住宿|宾馆)"),
                    re.compile(r"(酒店|住宿).{0,6}(价格|报价|星级|协议价)"),
                    re.compile(r"(有哪些|推荐|找).{0,6}(酒店|住宿)"),
                ),
                weak=(
                    re.compile(r"(酒店|住宿|宾馆|房型)"),
                    re.compile(r"(住哪|住一晚)"),
                ),
            ),
            TravelRule(
                "trip_planning",
                strong=(
                    re.compile(r"(安排|规划|制定|生成).{0,8}(出差|行程|差旅|方案)"),
                    re.compile(r"(出差|行程).{0,8}(方案|安排|规划)"),
                ),
                weak=(
                    re.compile(r"(出差|行程|差旅)"),
                    re.compile(r"(去.{1,8}(开会|拜访|培训|参展))"),
                ),
            ),
            TravelRule(
                "booking",
                strong=(
                    re.compile(r"(帮我|直接|现在).{0,4}(订|预订|出票|下单)"),
                    re.compile(r"(订|预订).{0,4}(机票|酒店|航班|房间)"),
                ),
                weak=(
                    re.compile(r"(订|预订|下单|出票)"),
                    re.compile(r"(就这个|确认预订)"),
                ),
            ),
            TravelRule(
                "policy",
                strong=(
                    re.compile(r"(差旅|出差).{0,6}(标准|制度|政策|规定)"),
                    re.compile(r"(住宿|酒店|舱位|机票).{0,6}(标准|上限|额度)"),
                    re.compile(r"(超标|超标准|能坐|能住)"),
                ),
                weak=(
                    re.compile(r"(标准|制度|政策|规定|额度)"),
                    re.compile(r"(报销|发票|凭证)"),
                ),
            ),
            TravelRule(
                "approval",
                strong=(
                    re.compile(r"(提交|发起|走).{0,6}(审批|申请)"),
                    re.compile(r"(审批|申请).{0,6}(流程|到哪|进度|需要谁)"),
                ),
                weak=(
                    re.compile(r"(审批|申请|领导|签字)"),
                ),
            ),
            TravelRule(
                "trip_status",
                strong=(
                    re.compile(r"(订单|行程单).{0,6}(状态|进度|到哪|查询)"),
                    re.compile(r"(我的|查).{0,4}(订单|行程单)"),
                    re.compile(r"(改签|退票|退订|取消订单)"),
                ),
                weak=(
                    re.compile(r"(订单|行程单|出票|退票|改签)"),
                ),
            ),
            TravelRule(
                "rag",
                strong=(
                    re.compile(r"(差旅|出差).{0,6}(管理办法|手册|文件|原文)"),
                    re.compile(r"(签证|目的地).{0,6}(要求|规定|安全)"),
                ),
                weak=(
                    re.compile(r"(在哪|原文|手册|文件)"),
                ),
            ),
            TravelRule(
                "general",
                strong=(
                    re.compile(r"^(你好|您好|在吗|hi|hello|谢谢|多谢)[!！。,.，?？\s]*$", re.IGNORECASE),
                ),
                weak=(
                    re.compile(r"^(嗯+|好的|收到)$"),
                ),
            ),
        ]

    def classify(self, text: str) -> Optional[tuple[str, float]]:
        normalized = text.strip()
        if not normalized:
            return None

        scores: dict[str, float] = {}
        strong_hits: dict[str, int] = {}
        weak_hits: dict[str, int] = {}

        for rule in self._rules:
            s = sum(1 for pat in rule.strong if pat.search(normalized))
            w = sum(1 for pat in rule.weak if pat.search(normalized))
            if s or w:
                strong_hits[rule.intent_slug] = s
                weak_hits[rule.intent_slug] = w
                scores[rule.intent_slug] = s * 2.0 + w * 1.0

        if not scores:
            return None

        best = max(scores, key=lambda slug: (scores[slug], _TIE_BREAK_PRIORITY.get(slug, 0)))
        s = strong_hits.get(best, 0)
        w = weak_hits.get(best, 0)
        if s:
            confidence = min(0.96, 0.86 + 0.04 * (s - 1))
        else:
            confidence = min(0.80, 0.68 + 0.06 * w)
        return best, confidence


@dataclass
class TravelIntentResult:
    intent: TravelIntent
    confidence: float
    slots: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.85


class TravelIntentRecognizer:
    """快慢双车道识别器。两路一致时抬升置信度，冲突时取更高一路并记录原因。"""

    def __init__(
        self,
        *,
        rule_engine: TravelRuleEngine | None = None,
        llm_classifier: LLMIntentClassifier | None = None,
        slow_lane_threshold: float = 0.85,
    ) -> None:
        self._rules = rule_engine or TravelRuleEngine()
        self._llm = llm_classifier
        self._threshold = slow_lane_threshold

    @classmethod
    def with_llm(
        cls,
        llm_client: Any,
        *,
        slow_lane_threshold: float = 0.85,
        model: str | None = None,
    ) -> "TravelIntentRecognizer":
        from app.core.intent.recognizer import StructuredLLMBridge

        bridge = StructuredLLMBridge(llm_client, model=model)
        classifier = LLMIntentClassifier(
            bridge,
            allowed_slugs=_DEFAULT_ALLOWED,
            system_prompt=TRAVEL_CLASSIFIER_SYSTEM,
        )
        return cls(llm_classifier=classifier, slow_lane_threshold=slow_lane_threshold)

    def _slug_to_intent(self, slug: str) -> TravelIntent:
        return _SLUG_TO_ENUM.get(slug, TravelIntent.GENERAL)

    def _merge(
        self,
        fast: Optional[tuple[str, float]],
        slow: Optional[LLMClassification],
    ) -> TravelIntentResult:
        if fast is None and slow is None:
            return TravelIntentResult(
                TravelIntent.UNKNOWN, 0.35, metadata={"fast_lane": None, "slow_lane": None}
            )
        if slow is None and fast is not None:
            return TravelIntentResult(
                self._slug_to_intent(fast[0]),
                fast[1],
                metadata={"fast_lane": fast, "slow_lane": None, "merged": "fast_only"},
            )
        if fast is None and slow is not None:
            return TravelIntentResult(
                self._slug_to_intent(slow.intent_slug),
                slow.confidence,
                metadata={"fast_lane": None, "slow_lane": slow, "merged": "slow_only"},
            )
        assert fast is not None and slow is not None
        fast_intent = self._slug_to_intent(fast[0])
        slow_intent = self._slug_to_intent(slow.intent_slug)
        if fast_intent is slow_intent:
            return TravelIntentResult(
                fast_intent,
                min(1.0, (fast[1] + slow.confidence) / 2 + 0.05),
                metadata={"fast_lane": fast, "slow_lane": slow, "merged": "agree"},
            )
        if fast[1] >= slow.confidence + 0.05:
            return TravelIntentResult(
                fast_intent,
                fast[1],
                metadata={"fast_lane": fast, "slow_lane": slow, "merged": "prefer_fast"},
            )
        return TravelIntentResult(
            slow_intent,
            slow.confidence,
            metadata={"fast_lane": fast, "slow_lane": slow, "merged": "prefer_slow"},
        )

    async def recognize(self, text: str) -> TravelIntentResult:
        fast = self._rules.classify(text)
        need_slow = self._llm is not None and (fast is None or fast[1] < self._threshold)
        slow: LLMClassification | None = None
        if need_slow:
            slow = await self._llm.classify(text)
        if not need_slow and fast is not None:
            return TravelIntentResult(
                self._slug_to_intent(fast[0]),
                fast[1],
                metadata={"fast_lane": fast, "slow_lane": None, "merged": "fast_only"},
            )
        return self._merge(fast, slow)


__all__ = [
    "TRAVEL_CLASSIFIER_SYSTEM",
    "TravelIntent",
    "TravelIntentRecognizer",
    "TravelIntentResult",
    "TravelRuleEngine",
]
