"""
规则引擎（快车道）- 基于关键词与正则的意图快速匹配。

适用于高频、表述相对固定的采购审批场景意图。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RuleSpec:
    """单条规则：意图 slug、权重、匹配模式。"""

    intent_slug: str
    weight: float
    patterns: tuple[re.Pattern[str], ...]


_TIE_BREAK_PRIORITY: dict[str, int] = {
    "policy": 100,
    "requisition_planning": 92,
    "application": 90,
    "approval": 88,
    "supplier_search": 74,
    "catalog_search": 72,
    "info_query": 60,
    "rag": 55,
}


class RuleEngine:
    """
    基于正则与关键词的规则分类器。

    匹配策略：遍历规则，对每条规则累计命中次数加权，取得分最高意图；
    分数相同时按业务优先级打破平局（如「审批政策」优先于单独「采购」词）。
    """

    def __init__(self) -> None:
        self._rules: list[RuleSpec] = [
            RuleSpec(
                "supplier_search",
                1.0,
                (
                    re.compile(r"(供应商|厂商|服务商|vendor)"),
                    re.compile(r"(找|查|推荐).{0,6}(供应商|厂商)"),
                ),
            ),
            RuleSpec(
                "catalog_search",
                1.0,
                (
                    re.compile(r"(商品|物料|设备|软件|采购目录|catalog)"),
                    re.compile(r"(查|搜|看看).{0,6}(报价|价格|目录|sku)"),
                    re.compile(r"(报价|价格).{0,8}(笔记本|电脑|软件|设备|物料)"),
                ),
            ),
            RuleSpec(
                "requisition_planning",
                1.0,
                (
                    re.compile(r"(规划|生成|整理|制定).{0,8}(采购|请购|申请单|方案)"),
                    re.compile(r"(采购|买).{0,6}(一批|一套|一些|申请)"),
                    re.compile(r"(请购单|采购单).{0,4}(草稿|方案|模板)"),
                ),
            ),
            RuleSpec(
                "application",
                1.0,
                (
                    re.compile(r"(提|提交|发起).{0,6}(申请|审批|报备)"),
                    re.compile(r"(采购|请购).{0,6}(申请|审批)"),
                ),
            ),
            RuleSpec(
                "policy",
                1.0,
                (
                    re.compile(r"(采购制度|审批制度|标准|额度|政策|合规)"),
                    re.compile(r"(超标|违规|不允许)"),
                ),
            ),
            RuleSpec(
                "approval",
                1.0,
                (
                    re.compile(r"(审批流|审批状态|审批进度|通过)"),
                    re.compile(r"(提交|进入).{0,6}(审批|流程)"),
                ),
            ),
            RuleSpec(
                "info_query",
                0.9,
                (
                    re.compile(r"(查询|查一下|看看).{0,8}(报价|供应商|政策|流程)"),
                    re.compile(r"(几点|多少钱|多久|在哪)"),
                ),
            ),
            RuleSpec(
                "rag",
                0.85,
                (
                    re.compile(r"(公司|内部).{0,6}(规定|制度|手册)"),
                    re.compile(r"(根据|依据).{0,6}(文档|知识)"),
                ),
            ),
        ]

    def classify(self, text: str) -> Optional[tuple[str, float]]:
        """
        对输入文本进行规则匹配。

        Returns:
            (意图 slug, 规则置信度 0~1)；无可信匹配时返回 None。
        """
        normalized = text.strip()
        if not normalized:
            return None

        scores: dict[str, float] = {}
        for rule in self._rules:
            hits = 0
            for pat in rule.patterns:
                if pat.search(normalized):
                    hits += 1
            if hits:
                scores[rule.intent_slug] = scores.get(rule.intent_slug, 0.0) + hits * rule.weight

        if not scores:
            return None

        best_slug = max(
            scores,
            key=lambda s: (scores[s], _TIE_BREAK_PRIORITY.get(s, 0)),
        )
        best_rule = next(r for r in self._rules if r.intent_slug == best_slug)
        hits = sum(1 for pat in best_rule.patterns if pat.search(normalized))
        confidence = min(1.0, 0.82 + 0.06 * max(0, hits - 1))
        return best_slug, confidence
