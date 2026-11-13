"""降级路径：LLM 不可用时给出确定性答复而不是 500。

采购场景恰好有可用的确定性兜底：`RuleEngine` 是纯正则，制度阈值和审批链是纯函数。模型挂了
仍然可以回答「8000 元以上软件采购需三家报价 + 信息安全评估」并给出审批路径，这比返回 500
有用得多。降级状态会显式标注，避免用户误以为拿到的是完整答复。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from app.core.intent.recognizer import ProcurementIntent
from app.core.tools.policy_query import PolicyQuery, query_procurement_policy
from app.core.tools.spec import ToolContext
from app.domain.procurement.models import PurchaseCategory

_DEGRADED_PREFIX = "⚠️ 当前模型服务不可用，以下为按公司制度规则直接计算的结果（非模型生成）。"

_CATEGORY_KEYWORDS: tuple[tuple[str, PurchaseCategory], ...] = (
    ("软件", PurchaseCategory.SOFTWARE),
    ("saas", PurchaseCategory.SOFTWARE),
    ("订阅", PurchaseCategory.SOFTWARE),
    ("设备", PurchaseCategory.EQUIPMENT),
    ("电脑", PurchaseCategory.EQUIPMENT),
    ("笔记本", PurchaseCategory.EQUIPMENT),
    ("营销", PurchaseCategory.MARKETING_SERVICE),
    ("推广", PurchaseCategory.MARKETING_SERVICE),
    ("咨询", PurchaseCategory.PROFESSIONAL_SERVICE),
    ("培训", PurchaseCategory.TRAINING),
    ("办公", PurchaseCategory.OFFICE_SUPPLIES),
)


@dataclass
class DegradedReply:
    text: str
    reason: str


def _guess_category(text: str, ctx: ToolContext) -> PurchaseCategory:
    known = ctx.draft.value("category")
    if known:
        try:
            return PurchaseCategory(known)
        except ValueError:
            pass
    lowered = text.lower()
    for keyword, category in _CATEGORY_KEYWORDS:
        if keyword in lowered:
            return category
    return PurchaseCategory.OTHER


def _guess_amount(ctx: ToolContext) -> Decimal | None:
    snapshot = ctx.draft.last_requisition or {}
    for candidate in (snapshot.get("estimated_total_cny"), ctx.draft.value("unit_price_cny")):
        if candidate is None:
            continue
        try:
            return Decimal(str(candidate))
        except (InvalidOperation, ValueError):
            continue
    return None


async def build_degraded_reply(
    user_text: str,
    intent: ProcurementIntent,
    ctx: ToolContext,
    *,
    reason: str,
) -> DegradedReply:
    """在不调用 LLM 的前提下拼出可用答复。"""
    lines: list[str] = [_DEGRADED_PREFIX]

    if intent in (ProcurementIntent.POLICY, ProcurementIntent.APPROVAL, ProcurementIntent.RAG):
        category = _guess_category(user_text, ctx)
        amount = _guess_amount(ctx) or Decimal("0")
        result = await query_procurement_policy(
            PolicyQuery(
                requester_level=ctx.profile.requester_level.value,
                category=category.value,
                amount_cny=amount,
                department=ctx.profile.department,
            ),
            policy=ctx.policy,
            today=ctx.today,
        )
        rules = result["rules"]
        lines.append(f"按品类「{category.value}」与金额 {amount} 元：")
        lines.append(
            "审批路径：" + " -> ".join(f"{s['role']}" for s in result["review_path"])
        )
        if rules.get("multi_quote_threshold_cny"):
            lines.append(f"多方报价阈值：{rules['multi_quote_threshold_cny']} 元")
        if rules.get("requires_security_review"):
            lines.append("该品类需信息安全评估。")
        if rules.get("requires_legal_review"):
            lines.append("该品类合同需法务复核。")
        if result["compliance_warnings"]:
            lines.append("合规提示：" + "；".join(result["compliance_warnings"]))
        lines.append(f"（依据制度版本 {ctx.policy_version}）")
    elif ctx.draft.missing_required():
        lines.append(ctx.draft.missing_prompt())
        lines.append("模型服务恢复后可继续生成请购单草稿。")
    else:
        lines.append("已记录你的请求。模型服务恢复后可继续处理；紧急采购请直接联系采购部。")

    if ctx.draft.summary_lines():
        lines.append("当前已收集的采购信息：")
        lines.extend(ctx.draft.summary_lines())

    return DegradedReply(text="\n".join(lines), reason=reason)
