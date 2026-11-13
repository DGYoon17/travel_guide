"""L2 槽位记忆：请购单草稿状态。

采购多轮对话的正确性不应该依赖模型的记忆力。这里把「已经问出来的信息」做成服务端状态：

- 每个槽位记录来源（用户说的 / 模型推断的 / 主数据带出的 / 工具算出的）与是否已确认；
- `confirmed=True` 的槽位禁止被模型静默改写，只能显式变更并留下 `overwrite_log`；
- 缺失的必填槽位由编排层生成确定性追问，保证"该问的一定问到"。

槽位值只允许 JSON 原生类型：这份状态要进 Redis，Decimal/date 必须先规范成字符串。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from decimal import Decimal
from typing import Any, Mapping

from pydantic import BaseModel, Field

from app.core.memory.slots import Slot, SlotChange, SlotSource, to_jsonable
from app.domain.procurement.models import PurchaseCategory, SupplierType, UrgencyLevel


#: 生成请购单必须齐备的槽位（主体身份字段不在此列，它们由服务端注入）。
REQUIRED_SLOTS: tuple[str, ...] = (
    "category",
    "item_name",
    "quantity",
    "unit_price_cny",
    "needed_by",
    "reason",
)

OPTIONAL_SLOTS: tuple[str, ...] = (
    "budget_code",
    "supplier_name",
    "supplier_type",
    "single_source",
    "urgency",
    "contract_months",
    "budget_ceiling_cny",
)

KNOWN_SLOTS: frozenset[str] = frozenset(REQUIRED_SLOTS + OPTIONAL_SLOTS)

#: 追问文案用的中文字段名，避免把 snake_case 字段名直接抛给用户。
SLOT_LABELS: dict[str, str] = {
    "category": "采购品类",
    "item_name": "采购物品/服务名称",
    "quantity": "数量",
    "unit_price_cny": "单价（元）",
    "needed_by": "需求到货/上线日期",
    "reason": "采购事由",
    "budget_code": "预算编码",
    "supplier_name": "供应商名称",
    "supplier_type": "供应商类型（既有/新供应商）",
    "single_source": "是否单一来源采购",
    "urgency": "紧急程度",
    "contract_months": "合同/订阅周期（月）",
    "budget_ceiling_cny": "预算上限（元）",
}

#: 给用户的取值提示，让追问一次到位。
SLOT_HINTS: dict[str, str] = {
    "category": "可选：" + "、".join(c.value for c in PurchaseCategory),
    "supplier_type": "可选：" + "、".join(s.value for s in SupplierType),
    "urgency": "可选：" + "、".join(u.value for u in UrgencyLevel),
    "needed_by": "格式 YYYY-MM-DD，或「下周五」这类相对表述",
}


def _jsonable(value: Any) -> Any:
    """兼容旧调用点的别名；实现已移到 `slots.to_jsonable`。"""
    return to_jsonable(value)


class RequisitionDraft(BaseModel):
    """一次采购需求的结构化草稿状态。"""

    draft_id: str = Field(default_factory=lambda: f"drf_{uuid.uuid4().hex[:12]}")
    session_id: str = ""
    tenant_id: str = "default"
    subject_id: str = ""
    version: int = 0
    slots: dict[str, Slot] = Field(default_factory=dict)
    rejected_changes: list[SlotChange] = Field(default_factory=list)
    overwrite_log: list[SlotChange] = Field(default_factory=list)
    #: 已生成的请购单摘要，供确认门展示。
    last_requisition: dict[str, Any] | None = None
    submitted_approval_no: str | None = None

    # ------------------------------------------------------------------ 读取
    def value(self, field: str) -> Any:
        slot = self.slots.get(field)
        return slot.value if slot else None

    def values(self) -> dict[str, Any]:
        return {k: s.value for k, s in self.slots.items() if s.value is not None}

    def missing_required(self) -> tuple[str, ...]:
        return tuple(f for f in REQUIRED_SLOTS if self.value(f) in (None, ""))

    def confirmed_fields(self) -> tuple[str, ...]:
        return tuple(sorted(f for f, s in self.slots.items() if s.confirmed))

    def is_complete(self) -> bool:
        return not self.missing_required()

    # ------------------------------------------------------------------ 写入
    def merge(
        self,
        updates: Mapping[str, Any],
        *,
        source: SlotSource,
        turn: int,
        allow_overwrite_confirmed: bool = False,
    ) -> tuple[tuple[str, ...], tuple[SlotChange, ...]]:
        """合并槽位更新。

        返回 `(已应用字段, 被拒绝的变更)`。已确认字段默认不可被覆盖——这是防止模型在后续
        轮次里把用户确认过的金额或日期"顺手改掉"的关键约束。
        """
        applied: list[str] = []
        rejected: list[SlotChange] = []

        for field, raw in updates.items():
            if field not in KNOWN_SLOTS:
                continue
            new_value = _jsonable(raw)
            if new_value in (None, ""):
                continue

            existing = self.slots.get(field)
            if existing is not None and existing.value == new_value:
                continue

            if existing is not None and existing.confirmed and not allow_overwrite_confirmed:
                change = SlotChange(
                    field=field,
                    old_value=existing.value,
                    new_value=new_value,
                    reason="字段已由用户确认，需用户显式同意才能修改",
                )
                rejected.append(change)
                self.rejected_changes.append(change)
                continue

            if existing is not None and existing.value is not None:
                self.overwrite_log.append(
                    SlotChange(
                        field=field,
                        old_value=existing.value,
                        new_value=new_value,
                        reason=f"由 {source.value} 更新",
                    )
                )

            self.slots[field] = Slot(
                value=new_value,
                source=source,
                confirmed=False,
                turn=turn,
            )
            applied.append(field)

        if applied:
            self.version += 1
        return tuple(applied), tuple(rejected)

    def confirm(self, fields: tuple[str, ...] | None = None) -> tuple[str, ...]:
        """标记字段为用户已确认。`fields=None` 表示确认当前全部已填槽位。"""
        targets = fields if fields is not None else tuple(self.slots.keys())
        changed: list[str] = []
        for field in targets:
            slot = self.slots.get(field)
            if slot is not None and not slot.confirmed:
                self.slots[field] = slot.model_copy(update={"confirmed": True})
                changed.append(field)
        if changed:
            self.version += 1
        return tuple(changed)

    def unconfirm(self, fields: tuple[str, ...]) -> None:
        """用户表示要改某项时解除确认锁。"""
        for field in fields:
            slot = self.slots.get(field)
            if slot is not None and slot.confirmed:
                self.slots[field] = slot.model_copy(update={"confirmed": False})
                self.version += 1

    # ------------------------------------------------------------------ 派生
    def fingerprint(self) -> str:
        """内容指纹。用于幂等键，确保同一份草稿重复提交不会产生两张请购单。"""
        payload = json.dumps(
            {
                "tenant": self.tenant_id,
                "subject": self.subject_id,
                "slots": {k: self.slots[k].value for k in sorted(self.slots)},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def missing_prompt(self) -> str:
        """缺失必填项的确定性追问文案。"""
        missing = self.missing_required()
        if not missing:
            return ""
        lines = ["还需要以下信息才能生成请购单："]
        for field in missing:
            label = SLOT_LABELS.get(field, field)
            hint = SLOT_HINTS.get(field)
            lines.append(f"- {label}" + (f"（{hint}）" if hint else ""))
        return "\n".join(lines)

    def summary_lines(self) -> list[str]:
        """已知槽位的可读摘要，用于确认门与结构化摘要交叉校验。"""
        lines: list[str] = []
        for field in REQUIRED_SLOTS + OPTIONAL_SLOTS:
            slot = self.slots.get(field)
            if slot is None or slot.value in (None, ""):
                continue
            mark = "✓" if slot.confirmed else "·"
            lines.append(f"{mark} {SLOT_LABELS.get(field, field)}：{slot.value}")
        return lines

    def known_amounts(self) -> set[str]:
        """草稿中出现过的金额字符串，供摘要环节做数字比对（防止摘要幻觉出新数字）。"""
        out: set[str] = set()
        for field in ("unit_price_cny", "budget_ceiling_cny"):
            value = self.value(field)
            if value is not None:
                out.add(str(value))
                try:
                    out.add(str(int(Decimal(str(value)))))
                except Exception:  # noqa: BLE001
                    pass
        total = (self.last_requisition or {}).get("estimated_total_cny")
        if total is not None:
            out.add(str(total))
        return out
