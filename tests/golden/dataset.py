"""金标评测集。

只放**确定性**用例：意图规则、参数抽取、合规校验、审批路径。这些不需要调用模型，因此可以
直接作为 CI 门禁跑。需要模型参与的端到端质量评估应另建离线评测任务，不放在单元测试里。

新增用例请在这里追加，不要改阈值来让测试通过 —— 阈值下调必须单独说明理由。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

REFERENCE_TODAY = date(2026, 8, 20)  # 星期四


# --------------------------------------------------------------------- 意图分类


@dataclass(frozen=True)
class IntentCase:
    text: str
    expected: str


INTENT_CASES: tuple[IntentCase, ...] = (
    IntentCase("帮我查一下笔记本采购目录和报价", "catalog_search"),
    IntentCase("公司的采购审批政策和预算上限是多少", "policy"),
    IntentCase("这笔采购超标了吗", "policy"),
    IntentCase("帮我找几家做视频制作的供应商", "supplier_search"),
    IntentCase("推荐一下打印机耗材的厂商", "supplier_search"),
    IntentCase("帮我生成一份采购申请单草稿", "requisition_planning"),
    IntentCase("我们要采购一套设计协作软件", "requisition_planning"),
    IntentCase("请提交审批", "application"),
    IntentCase("审批进度到哪一步了", "approval"),
    IntentCase("公司内部的报销制度手册在哪", "rag"),
    IntentCase("你好", "general"),
)

#: 规则引擎（快车道）在金标集上的最低准确率。低于此值说明规则退化。
MIN_INTENT_ACCURACY = 0.80


# --------------------------------------------------------------------- 参数抽取


@dataclass(frozen=True)
class SlotCase:
    """模型可能给出的原始参数 → 期望落库的规范值。

    这是最关键的一类：金额或日期抽错会直接导致错误的审批链。
    """

    name: str
    raw: dict[str, Any]
    expected: dict[str, Any]


SLOT_CASES: tuple[SlotCase, ...] = (
    SlotCase(
        "iso_date_and_plain_amount",
        {"needed_by": "2026-09-15", "unit_price_cny": "15000"},
        {"needed_by": "2026-09-15", "unit_price_cny": "15000.00"},
    ),
    SlotCase(
        "relative_weekday",
        {"needed_by": "下周五", "unit_price_cny": "15000"},
        {"needed_by": "2026-08-28", "unit_price_cny": "15000.00"},
    ),
    SlotCase(
        "chinese_myriad_amount",
        {"needed_by": "明天", "unit_price_cny": "1.5万"},
        {"needed_by": "2026-08-21", "unit_price_cny": "15000.00"},
    ),
    SlotCase(
        "amount_with_unit_and_separator",
        {"needed_by": "本月底", "unit_price_cny": "18,800元"},
        {"needed_by": "2026-08-31", "unit_price_cny": "18800.00"},
    ),
    SlotCase(
        "business_days",
        {"needed_by": "3个工作日后", "unit_price_cny": "8千"},
        {"needed_by": "2026-08-25", "unit_price_cny": "8000.00"},
    ),
    SlotCase(
        "month_day_rolls_forward",
        {"needed_by": "1月10日", "unit_price_cny": "6299"},
        {"needed_by": "2027-01-10", "unit_price_cny": "6299.00"},
    ),
)

MIN_SLOT_ACCURACY = 1.0  # 参数抽取不允许回退


# --------------------------------------------------------------------- 合规校验


@dataclass(frozen=True)
class ComplianceCase:
    """必须命中的合规提示。漏报（false negative）比误报危险得多。"""

    name: str
    category: str
    total_cny: str
    needed_in_days: int = 30
    supplier_type: str = "existing_vendor"
    single_source: bool = False
    contract_months: int = 1
    requester_level: str = "manager"
    must_warn_about: tuple[str, ...] = ()
    must_include_roles: tuple[str, ...] = ()
    must_exclude_roles: tuple[str, ...] = ()


COMPLIANCE_CASES: tuple[ComplianceCase, ...] = (
    ComplianceCase(
        "software_above_multi_quote_threshold",
        category="software",
        total_cny="18800",
        must_warn_about=("三家报价", "信息安全"),
        must_include_roles=("部门负责人", "预算负责人", "采购经理", "财务BP", "信息安全"),
    ),
    ComplianceCase(
        "new_vendor_requires_onboarding",
        category="equipment",
        total_cny="12000",
        supplier_type="new_vendor",
        must_warn_about=("新供应商",),
        must_include_roles=("采购经理",),
    ),
    ComplianceCase(
        "sole_source_above_threshold_needs_reason",
        category="equipment",
        total_cny="20000",
        single_source=True,
        must_warn_about=("单一来源",),
    ),
    ComplianceCase(
        "insufficient_lead_time",
        category="office_supplies",
        total_cny="2000",
        needed_in_days=1,
        must_warn_about=("提前",),
        must_exclude_roles=("预算负责人",),
    ),
    ComplianceCase(
        "long_contract_needs_legal",
        category="professional_service",
        total_cny="30000",
        contract_months=24,
        must_warn_about=("法务",),
        must_include_roles=("法务",),
    ),
    ComplianceCase(
        "small_office_purchase_stays_light",
        category="office_supplies",
        total_cny="800",
        must_include_roles=("部门负责人", "采购经理"),
        must_exclude_roles=("预算负责人", "财务BP", "分管高管", "信息安全"),
    ),
)

MIN_COMPLIANCE_RECALL = 1.0  # 合规漏报零容忍
MIN_APPROVAL_PATH_ACCURACY = 1.0


@dataclass
class MetricReport:
    """评测结果，便于在 CI 输出里直接看到退化点。"""

    name: str
    total: int = 0
    passed: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 1.0

    def record(self, ok: bool, detail: str = "") -> None:
        self.total += 1
        if ok:
            self.passed += 1
        elif detail:
            self.failures.append(detail)

    def summary(self) -> str:
        head = f"{self.name}: {self.passed}/{self.total} = {self.score:.2%}"
        if not self.failures:
            return head
        return head + "\n  - " + "\n  - ".join(self.failures[:10])
