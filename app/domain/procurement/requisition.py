from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from app.core.clock import at_business_time, business_now
from app.domain.procurement.models import (
    ApprovalStep,
    ProcurementLine,
    ProcurementPolicy,
    ProcurementRequest,
    PurchaseCategory,
    PurchaseRequisition,
)


def _local_now() -> datetime:
    return business_now()


def _normalized_amount(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def build_draft_requisition(request: ProcurementRequest) -> PurchaseRequisition:
    """
    根据采购申请生成请购单草稿。

    生产环境可替换为商品目录、供应商报价或 SRM 接口结果。
    """
    subtotal = _normalized_amount(request.unit_price_cny * request.quantity)
    line = ProcurementLine(
        line_id=str(uuid.uuid4()),
        item_name=request.item_name,
        category=request.category,
        quantity=request.quantity,
        unit_price_cny=_normalized_amount(request.unit_price_cny),
        subtotal_cny=subtotal,
        supplier_name=request.supplier_name,
    )
    title = f"{request.department} {request.item_name}采购申请"
    return PurchaseRequisition(
        requisition_id=str(uuid.uuid4()),
        request_id=request.request_id,
        title=title,
        department=request.department,
        line_items=[line],
        estimated_total_cny=subtotal,
        created_at=at_business_time(request.needed_by, 9, 0),
    )


def build_default_approval_path(
    policy: ProcurementPolicy,
    request: ProcurementRequest,
    total_amount_cny: Decimal,
) -> list[ApprovalStep]:
    rule = policy.approval_rules[request.requester_level]
    steps = [
        ApprovalStep(role="部门负责人", reason="确认采购必要性与需求真实性"),
    ]
    if total_amount_cny > policy.budget_owner_review_threshold_cny:
        steps.append(ApprovalStep(role="预算负责人", reason="校验预算编码与额度可用性"))
    steps.append(ApprovalStep(role="采购经理", reason="核验采购方式、询价与供应商选择"))
    if total_amount_cny > rule.finance_review_threshold_cny:
        steps.append(ApprovalStep(role="财务BP", reason="复核金额、税率与付款节奏"))
    if request.category is PurchaseCategory.SOFTWARE:
        steps.append(ApprovalStep(role="信息安全", reason="校验软件许可、数据权限与接入风险"))
    if request.contract_months >= 12 or request.category in (
        PurchaseCategory.MARKETING_SERVICE,
        PurchaseCategory.PROFESSIONAL_SERVICE,
    ):
        steps.append(ApprovalStep(role="法务", reason="审核合同条款与供应商责任边界"))
    if total_amount_cny > rule.executive_review_threshold_cny:
        steps.append(ApprovalStep(role="分管高管", reason="超阈值采购需高层批准"))
    return steps


def summarize_requisition_text(requisition: PurchaseRequisition) -> str:
    lines = [f"【{requisition.title}】", f"预估总额：{requisition.estimated_total_cny} CNY"]
    for i, line in enumerate(requisition.line_items, start=1):
        supplier = line.supplier_name or "待定供应商"
        lines.append(
            f"{i}. {line.item_name} x{line.quantity}，单价 {line.unit_price_cny} CNY，"
            f"小计 {line.subtotal_cny} CNY，供应商：{supplier}"
        )
    if requisition.approval_path:
        lines.append(
            "审批路径：" + " -> ".join(f"{step.role}（{step.reason}）" for step in requisition.approval_path)
        )
    if requisition.compliance_warnings:
        lines.append("合规提示：" + "；".join(requisition.compliance_warnings))
    return "\n".join(lines)
