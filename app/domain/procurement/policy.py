from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.domain.procurement.models import (
    ApprovalThresholdRule,
    CategoryRule,
    ProcurementPolicy,
    ProcurementRequest,
    PurchaseCategory,
    PurchaseRequisition,
    RequesterLevel,
    SupplierType,
)
from app.domain.procurement.requisition import build_default_approval_path


def default_procurement_policy(company_id: str = "default") -> ProcurementPolicy:
    return ProcurementPolicy(
        policy_id=f"PROC-{company_id}",
        company_id=company_id,
        effective_from=date.today(),
        approval_rules={
            RequesterLevel.STAFF: ApprovalThresholdRule(
                max_auto_approve_cny=Decimal("1000"),
                finance_review_threshold_cny=Decimal("5000"),
                executive_review_threshold_cny=Decimal("20000"),
            ),
            RequesterLevel.MANAGER: ApprovalThresholdRule(
                max_auto_approve_cny=Decimal("3000"),
                finance_review_threshold_cny=Decimal("8000"),
                executive_review_threshold_cny=Decimal("30000"),
            ),
            RequesterLevel.DIRECTOR: ApprovalThresholdRule(
                max_auto_approve_cny=Decimal("5000"),
                finance_review_threshold_cny=Decimal("12000"),
                executive_review_threshold_cny=Decimal("50000"),
            ),
            RequesterLevel.EXECUTIVE: ApprovalThresholdRule(
                max_auto_approve_cny=Decimal("10000"),
                finance_review_threshold_cny=Decimal("20000"),
                executive_review_threshold_cny=Decimal("80000"),
            ),
        },
        category_rules={
            PurchaseCategory.OFFICE_SUPPLIES: CategoryRule(
                need_multi_quote_above_cny=Decimal("5000")
            ),
            PurchaseCategory.SOFTWARE: CategoryRule(
                need_multi_quote_above_cny=Decimal("8000"),
                requires_security_review=True,
                requires_legal_review=True,
            ),
            PurchaseCategory.EQUIPMENT: CategoryRule(
                need_multi_quote_above_cny=Decimal("10000")
            ),
            PurchaseCategory.MARKETING_SERVICE: CategoryRule(
                need_multi_quote_above_cny=Decimal("15000"),
                requires_legal_review=True,
            ),
            PurchaseCategory.PROFESSIONAL_SERVICE: CategoryRule(
                need_multi_quote_above_cny=Decimal("12000"),
                requires_legal_review=True,
            ),
            PurchaseCategory.TRAINING: CategoryRule(
                need_multi_quote_above_cny=Decimal("6000")
            ),
            PurchaseCategory.OTHER: CategoryRule(
                need_multi_quote_above_cny=Decimal("5000")
            ),
        },
        urgent_lead_days_min=3,
        budget_owner_review_threshold_cny=Decimal("3000"),
        new_vendor_review_required=True,
        sole_source_reason_required_above_cny=Decimal("8000"),
    )


def evaluate_procurement_compliance(
    policy: ProcurementPolicy,
    request: ProcurementRequest,
    requisition: PurchaseRequisition,
) -> list[str]:
    warnings: list[str] = []
    if request.budget_ceiling_cny and requisition.estimated_total_cny > request.budget_ceiling_cny:
        warnings.append(
            f"采购预估 {requisition.estimated_total_cny} CNY 超过预算上限 {request.budget_ceiling_cny} CNY。"
        )
    delta = (request.needed_by - date.today()).days
    if delta < policy.urgent_lead_days_min:
        warnings.append(
            f"距需求日期仅 {delta} 天，低于建议提前 {policy.urgent_lead_days_min} 天发起采购的要求。"
        )
    category_rule = policy.category_rules.get(request.category)
    if category_rule and requisition.estimated_total_cny >= category_rule.need_multi_quote_above_cny:
        warnings.append(
            f"{request.category.value} 类采购金额达到 {category_rule.need_multi_quote_above_cny} CNY，需提供至少三家报价。"
        )
    if request.single_source and requisition.estimated_total_cny >= policy.sole_source_reason_required_above_cny:
        warnings.append(
            f"单一来源采购金额达到 {policy.sole_source_reason_required_above_cny} CNY，需补充唯一供应商说明。"
        )
    if policy.new_vendor_review_required and request.supplier_type is SupplierType.NEW:
        warnings.append("新供应商需先完成准入、税务与收款信息校验。")
    if category_rule and category_rule.requires_security_review:
        warnings.append("该品类需信息安全评估后方可签约或上线。")
    if category_rule and category_rule.requires_legal_review and request.contract_months >= 12:
        warnings.append("合同周期较长，需法务复核关键条款与续约条件。")
    return warnings


def apply_policy_to_requisition(
    policy: ProcurementPolicy,
    request: ProcurementRequest,
    requisition: PurchaseRequisition,
) -> PurchaseRequisition:
    warnings = list(requisition.compliance_warnings)
    warnings.extend(evaluate_procurement_compliance(policy, request, requisition))
    approval_path = build_default_approval_path(policy, request, requisition.estimated_total_cny)
    return requisition.model_copy(
        update={
            "compliance_warnings": warnings,
            "approval_path": approval_path,
        }
    )
