"""采购制度查询：委托给领域层，避免审批路径出现第二份真相。

改造前这里自带一套阈值和 `review_path` 拼装逻辑，与
`app/domain/procurement/policy.py` + `requisition.build_default_approval_path` 并存。
两处都算审批链，改一处另一处静默漂移 —— 用户拿到的"审批路径"取决于走了哪条代码路径。
现在统一由领域层计算，本模块只做入参归一与结果整形。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from app.domain.procurement.models import (
    ProcurementPolicy,
    ProcurementRequest,
    PurchaseCategory,
    RequesterLevel,
    SupplierType,
    UrgencyLevel,
)
from app.domain.procurement.policy import (
    default_procurement_policy,
    evaluate_procurement_compliance,
)
from app.domain.procurement.requisition import build_default_approval_path, build_draft_requisition


@dataclass
class PolicyQuery:
    requester_level: str
    category: str
    # 金额用 Decimal 而非 float：改造前这里是 float，导致
    # Decimal → float → Decimal 的往返，既丢精度也让 str() 输出不稳定（15000.00 → 15000.0）。
    amount_cny: Decimal = Decimal("0")
    supplier_type: str = "existing_vendor"
    single_source: bool = False
    contract_months: int = 1
    needed_by: date | None = None
    department: str = "未指定部门"

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "PolicyQuery":
        needed = data.get("needed_by")
        if isinstance(needed, str):
            needed = date.fromisoformat(needed)
        return cls(
            requester_level=str(data["requester_level"]),
            category=str(data["category"]),
            amount_cny=Decimal(str(data.get("amount_cny", 0))),
            supplier_type=str(data.get("supplier_type", "existing_vendor")),
            single_source=bool(data.get("single_source", False)),
            contract_months=int(data.get("contract_months", 1)),
            needed_by=needed,
            department=str(data.get("department", "未指定部门")),
        )


def _to_request(q: PolicyQuery, *, today: date) -> ProcurementRequest:
    amount = q.amount_cny
    return ProcurementRequest(
        request_id=str(uuid.uuid4()),
        requester_id="policy-preview",
        requester_level=RequesterLevel(q.requester_level),
        department=q.department,
        category=PurchaseCategory(q.category),
        item_name="制度校验预览项",
        quantity=1,
        unit_price_cny=amount,
        needed_by=q.needed_by or (today + timedelta(days=14)),
        reason="采购制度与审批路径预校验",
        supplier_type=SupplierType(q.supplier_type),
        single_source=q.single_source,
        urgency=UrgencyLevel.NORMAL,
        contract_months=q.contract_months,
    )


def category_thresholds(policy: ProcurementPolicy, category: PurchaseCategory) -> dict[str, Any]:
    """暴露该品类的硬性阈值，供回答里标注依据。"""
    rule = policy.category_rules.get(category)
    return {
        "multi_quote_threshold_cny": str(rule.need_multi_quote_above_cny) if rule else None,
        "requires_security_review": bool(rule and rule.requires_security_review),
        "requires_legal_review": bool(rule and rule.requires_legal_review),
        "budget_owner_review_threshold_cny": str(policy.budget_owner_review_threshold_cny),
        "sole_source_reason_required_above_cny": str(policy.sole_source_reason_required_above_cny),
        "urgent_lead_days_min": policy.urgent_lead_days_min,
    }


async def query_procurement_policy(
    q: PolicyQuery,
    *,
    policy: ProcurementPolicy | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """返回审批路径、硬性阈值与合规提示。审批路径与请购单生成走同一函数。"""
    pol = policy or default_procurement_policy()
    ref_today = today or date.today()
    request = _to_request(q, today=ref_today)
    amount = q.amount_cny

    requisition = build_draft_requisition(request).model_copy(update={"estimated_total_cny": amount})
    approval_path = build_default_approval_path(pol, request, amount)
    warnings = evaluate_procurement_compliance(pol, request, requisition)

    return {
        "requester_level": q.requester_level,
        "category": q.category,
        "amount_cny": str(amount),
        "supplier_type": q.supplier_type,
        "policy_id": pol.policy_id,
        "rules": category_thresholds(pol, PurchaseCategory(q.category)),
        "review_path": [{"role": s.role, "reason": s.reason} for s in approval_path],
        "compliance_warnings": warnings,
    }


async def policy_query(**kwargs: Any) -> dict[str, Any]:
    q = PolicyQuery.model_validate(kwargs)
    return await query_procurement_policy(q)
