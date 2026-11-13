"""金标评测：意图准确率、参数抽取准确率、合规召回率、审批路径正确率。

这四项是采购 Agent 最容易在改 prompt / 改规则时静默退化的地方，因此做成 CI 门禁：
分数低于 `tests/golden/dataset.py` 里的阈值就直接失败，并打印退化的具体用例。
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from app.core.intent.rule_engine import RuleEngine
from app.core.tools.coercion import parse_cn_date, parse_cny
from app.core.tools.policy_query import PolicyQuery, query_procurement_policy
from app.domain.procurement.policy import default_procurement_policy
from tests.golden.dataset import (
    COMPLIANCE_CASES,
    INTENT_CASES,
    MIN_APPROVAL_PATH_ACCURACY,
    MIN_COMPLIANCE_RECALL,
    MIN_INTENT_ACCURACY,
    MIN_SLOT_ACCURACY,
    REFERENCE_TODAY,
    SLOT_CASES,
    MetricReport,
)


def test_intent_rule_accuracy_meets_threshold() -> None:
    engine = RuleEngine()
    report = MetricReport("intent_accuracy")
    for case in INTENT_CASES:
        result = engine.classify(case.text)
        actual = result[0] if result else "general"
        report.record(
            actual == case.expected,
            f"「{case.text}」期望 {case.expected}，实际 {actual}",
        )
    assert report.score >= MIN_INTENT_ACCURACY, report.summary()


def test_slot_extraction_accuracy_is_perfect() -> None:
    """金额/日期抽错直接导致错误审批链，因此这一项不允许有失败用例。"""
    report = MetricReport("slot_extraction_accuracy")
    for case in SLOT_CASES:
        try:
            parsed = {
                "needed_by": parse_cn_date(
                    case.raw["needed_by"], field="needed_by", today=REFERENCE_TODAY
                ).isoformat(),
                "unit_price_cny": str(
                    parse_cny(case.raw["unit_price_cny"], field="unit_price_cny")
                ),
            }
            ok = parsed == case.expected
            detail = f"{case.name}: 期望 {case.expected}，实际 {parsed}"
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{case.name}: 抛出 {type(exc).__name__}: {exc}"
        report.record(ok, detail)
    assert report.score >= MIN_SLOT_ACCURACY, report.summary()


async def _evaluate(case) -> tuple[list[str], list[str]]:  # noqa: ANN001
    policy = default_procurement_policy()
    result = await query_procurement_policy(
        PolicyQuery(
            requester_level=case.requester_level,
            category=case.category,
            amount_cny=Decimal(case.total_cny),
            supplier_type=case.supplier_type,
            single_source=case.single_source,
            contract_months=case.contract_months,
            needed_by=REFERENCE_TODAY + timedelta(days=case.needed_in_days),
            department="市场部",
        ),
        policy=policy,
        today=REFERENCE_TODAY,
    )
    warnings = list(result["compliance_warnings"])
    roles = [step["role"] for step in result["review_path"]]
    return warnings, roles


async def test_compliance_recall_has_no_false_negatives() -> None:
    report = MetricReport("compliance_recall")
    for case in COMPLIANCE_CASES:
        warnings, _ = await _evaluate(case)
        blob = "；".join(warnings)
        for keyword in case.must_warn_about:
            report.record(
                keyword in blob,
                f"{case.name}: 漏报「{keyword}」，实际提示={warnings}",
            )
    assert report.score >= MIN_COMPLIANCE_RECALL, report.summary()


async def test_approval_path_accuracy() -> None:
    report = MetricReport("approval_path_accuracy")
    for case in COMPLIANCE_CASES:
        _, roles = await _evaluate(case)
        for role in case.must_include_roles:
            report.record(role in roles, f"{case.name}: 审批链缺少「{role}」，实际={roles}")
        for role in case.must_exclude_roles:
            report.record(
                role not in roles, f"{case.name}: 审批链多了「{role}」，实际={roles}"
            )
    assert report.score >= MIN_APPROVAL_PATH_ACCURACY, report.summary()


async def test_approval_path_is_monotonic_in_amount() -> None:
    """金额变大时审批节点只能增加不能减少 —— 阈值配错时这条最先炸。"""
    policy = default_procurement_policy()
    previous: set[str] = set()
    for amount in (500, 3500, 9000, 25000, 60000):
        result = await query_procurement_policy(
            PolicyQuery(
                requester_level="manager",
                category="equipment",
                amount_cny=Decimal(amount),
                needed_by=REFERENCE_TODAY + timedelta(days=30),
            ),
            policy=policy,
            today=REFERENCE_TODAY,
        )
        roles = {step["role"] for step in result["review_path"]}
        assert previous <= roles, f"金额 {amount} 时审批节点反而减少：{previous - roles}"
        previous = roles


def test_golden_dataset_is_not_silently_shrunk() -> None:
    """防止有人为了让门禁通过而删用例。"""
    assert len(INTENT_CASES) >= 11
    assert len(SLOT_CASES) >= 6
    assert len(COMPLIANCE_CASES) >= 6
