"""L2 差旅槽位记忆：一次出差的草稿状态。

复用采购域验证过的三条约束：

- 槽位带来源与确认标记，已确认字段禁止模型静默改写；
- 槽位值只允许 JSON 原生类型，可安全写入 Redis；
- 缺失必填项由编排层确定性追问，不依赖模型记忆。

差旅特有的部分是 `last_plan` / `selected_flight` / `selected_hotel`：
它们不是槽位，而是工具产出的结构化结果，预订确认门要拿它做快照与二次校验。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date
from decimal import Decimal
from typing import Any, Mapping

from pydantic import BaseModel, Field

from app.core.memory.profile import SubjectProfile
from app.core.memory.slots import Slot, SlotChange, SlotSource, to_jsonable
from app.core.tools.coercion import parse_cn_date, parse_cny
from app.domain.travel.models import (
    CabinClass,
    TravelerLevel,
    TripPurpose,
    TripRequest,
    TripType,
)


REQUIRED_SLOTS: tuple[str, ...] = (
    "origin_city",
    "destination_city",
    "depart_date",
    "purpose",
)

OPTIONAL_SLOTS: tuple[str, ...] = (
    "return_date",
    "trip_type",
    "travelers",
    "cabin_class",
    "hotel_needed",
    "hotel_star",
    "budget_cny",
    "cost_center",
    "project_code",
    "international",
    "passport_ready",
    "estimated_flight_minutes",
    "reason",
)

KNOWN_SLOTS: frozenset[str] = frozenset(REQUIRED_SLOTS + OPTIONAL_SLOTS)

SLOT_LABELS: dict[str, str] = {
    "origin_city": "出发城市",
    "destination_city": "目的地城市",
    "depart_date": "出发日期",
    "return_date": "返程日期",
    "purpose": "出差事由",
    "trip_type": "行程类型（单程/往返/多程）",
    "travelers": "出行人数",
    "cabin_class": "舱位",
    "hotel_needed": "是否需要住宿",
    "hotel_star": "酒店星级",
    "budget_cny": "预算（元）",
    "cost_center": "成本中心",
    "project_code": "项目号",
    "international": "是否国际行程",
    "passport_ready": "护照/签证是否就绪",
    "estimated_flight_minutes": "预计飞行时长（分钟）",
    "reason": "补充说明",
}

SLOT_HINTS: dict[str, str] = {
    "depart_date": "格式 YYYY-MM-DD，或「下周五」「3 天后」",
    "return_date": "格式 YYYY-MM-DD，或「下周日」",
    "cabin_class": "可选：" + "、".join(c.value for c in CabinClass),
    "trip_type": "可选：" + "、".join(t.value for t in TripType),
    "purpose": "如：拜访客户、内部会议、培训、行业会议",
    "estimated_flight_minutes": "整数分钟，如 150（约 2.5 小时）",
}

_PURPOSE_ALIASES: dict[str, TripPurpose] = {
    "客户": TripPurpose.CLIENT,
    "拜访": TripPurpose.CLIENT,
    "商务": TripPurpose.CLIENT,
    "内部": TripPurpose.INTERNAL,
    "会议": TripPurpose.CONFERENCE,
    "培训": TripPurpose.TRAINING,
    "行业": TripPurpose.CONFERENCE,
}


def _coerce_purpose(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return TripPurpose.INTERNAL.value
    try:
        return TripPurpose(text).value
    except ValueError:
        pass
    for key, purpose in _PURPOSE_ALIASES.items():
        if key in text:
            return purpose.value
    return TripPurpose.OTHER.value


def _as_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "是", "需要", "要"}


class TripDraft(BaseModel):
    """一次出差的完整草稿状态。"""

    draft_id: str = Field(default_factory=lambda: f"trip_{uuid.uuid4().hex[:12]}")
    session_id: str = ""
    tenant_id: str = "default"
    subject_id: str = ""
    version: int = 0
    slots: dict[str, Slot] = Field(default_factory=dict)
    rejected_changes: list[SlotChange] = Field(default_factory=list)
    overwrite_log: list[SlotChange] = Field(default_factory=list)
    #: `plan_trip` 产出的行程方案快照（含 plan_id / segments / total / 审批链）。
    last_plan: dict[str, Any] | None = None
    selected_flight: dict[str, Any] | None = None
    selected_hotel: dict[str, Any] | None = None
    booking_ids: list[str] = Field(default_factory=list)
    submitted_approval_no: str | None = None
    #: `build_travel_request` 产出的出差申请单草稿。
    last_request: dict[str, Any] | None = None

    # ------------------------------------------------------------------ 读取
    def value(self, field: str) -> Any:
        slot = self.slots.get(field)
        return slot.value if slot else None

    def values(self) -> dict[str, Any]:
        return {k: s.value for k, s in self.slots.items() if s.value is not None}

    def missing_required(self) -> tuple[str, ...]:
        missing = [f for f in REQUIRED_SLOTS if self.value(f) in (None, "")]
        if self.value("trip_type") == TripType.ROUND_TRIP.value and not self.value("return_date"):
            missing.append("return_date")
        return tuple(missing)

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
        applied: list[str] = []
        rejected: list[SlotChange] = []

        for field, raw in updates.items():
            if field not in KNOWN_SLOTS:
                continue
            value = self._normalize(field, raw)
            if value in (None, ""):
                continue

            existing = self.slots.get(field)
            if existing is not None and existing.value == value:
                continue
            if existing is not None and existing.confirmed and not allow_overwrite_confirmed:
                change = SlotChange(
                    field=field,
                    old_value=existing.value,
                    new_value=value,
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
                        new_value=value,
                        reason=f"由 {source.value} 更新",
                    )
                )
            self.slots[field] = Slot(value=value, source=source, confirmed=False, turn=turn)
            applied.append(field)

        if applied:
            self.version += 1
        return tuple(applied), tuple(rejected)

    def _normalize(self, field: str, raw: Any) -> Any:
        if field in {"depart_date", "return_date"}:
            try:
                parsed = parse_cn_date(raw, field=field)
                # 槽位必须保持 JSON 原生类型，否则 Redis 持久化与指纹都会炸。
                return parsed.isoformat() if parsed else None
            except Exception:  # noqa: BLE001 - 解析失败由工具层给出字段级错误
                return None
        if field == "purpose":
            return _coerce_purpose(raw)
        if field in {"hotel_needed", "international", "passport_ready"}:
            return _as_bool(raw)
        if field in {"travelers", "hotel_star", "estimated_flight_minutes"}:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None
        if field == "budget_cny":
            try:
                return str(parse_cny(raw, field="budget_cny"))
            except Exception:  # noqa: BLE001
                return None
        return to_jsonable(raw)

    def confirm(self, fields: tuple[str, ...] | None = None) -> tuple[str, ...]:
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
        for field in fields:
            slot = self.slots.get(field)
            if slot is not None and slot.confirmed:
                self.slots[field] = slot.model_copy(update={"confirmed": False})
                self.version += 1

    # ------------------------------------------------------------------ 派生
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "tenant": self.tenant_id,
                "subject": self.subject_id,
                "slots": {k: self.slots[k].value for k in sorted(self.slots)},
                "plan": (self.last_plan or {}).get("plan_id"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def missing_prompt(self) -> str:
        missing = self.missing_required()
        if not missing:
            return ""
        lines = ["还需要以下信息才能规划行程："]
        for field in missing:
            label = SLOT_LABELS.get(field, field)
            hint = SLOT_HINTS.get(field)
            lines.append(f"- {label}" + (f"（{hint}）" if hint else ""))
        return "\n".join(lines)

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        for field in REQUIRED_SLOTS + OPTIONAL_SLOTS:
            slot = self.slots.get(field)
            if slot is None or slot.value in (None, ""):
                continue
            mark = "✓" if slot.confirmed else "·"
            lines.append(f"{mark} {SLOT_LABELS.get(field, field)}：{slot.value}")
        return lines

    def known_amounts(self) -> set[str]:
        out: set[str] = set()
        budget = self.value("budget_cny")
        if budget is not None:
            out.add(str(budget))
        for offer in (self.selected_flight, self.selected_hotel):
            if offer and offer.get("price_cny") is not None:
                out.add(str(offer["price_cny"]))
        total = (self.last_plan or {}).get("total_cny")
        if total is not None:
            out.add(str(total))
        return out

    # ------------------------------------------------------------------ 组装
    def to_trip_request(self, profile: SubjectProfile) -> TripRequest:
        """把槽位 + 画像组装成领域请求。主体字段只来自 profile。"""
        depart = parse_cn_date(self.value("depart_date"), field="depart_date")
        if depart is None:
            raise ValueError("depart_date is required")
        return_raw = self.value("return_date")
        return_date = parse_cn_date(return_raw, field="return_date") if return_raw else None

        trip_type_raw = self.value("trip_type")
        if trip_type_raw:
            trip_type = TripType(str(trip_type_raw))
        else:
            trip_type = TripType.ROUND_TRIP if return_date else TripType.ONE_WAY

        cabin_raw = self.value("cabin_class") or CabinClass.ECONOMY.value
        budget_raw = self.value("budget_cny")

        return TripRequest(
            trip_id=self.draft_id,
            traveler_id=profile.subject_id,
            traveler_level=TravelerLevel(profile.requester_level.value),
            department=profile.department,
            tenant_id=profile.tenant_id,
            origin_city=str(self.value("origin_city")),
            destination_city=str(self.value("destination_city")),
            depart_date=depart,
            return_date=return_date,
            trip_type=trip_type,
            purpose=TripPurpose(_coerce_purpose(self.value("purpose"))),
            travelers=int(self.value("travelers") or 1),
            cabin_class=CabinClass(str(cabin_raw)),
            hotel_needed=_as_bool(self.value("hotel_needed")),
            hotel_star=int(self.value("hotel_star") or 4),
            budget_cny=Decimal(str(budget_raw)) if budget_raw not in (None, "") else None,
            cost_center=str(self.value("cost_center") or profile.cost_center or "") or None,
            project_code=str(self.value("project_code") or "") or None,
            international=_as_bool(self.value("international")),
            passport_ready=_as_bool(self.value("passport_ready")) if self.value("passport_ready") is not None else True,
            estimated_flight_minutes=int(self.value("estimated_flight_minutes") or 150),
            reason=str(self.value("reason") or self.value("purpose") or ""),
        )


def new_trip_draft(*, session_id: str, tenant_id: str, subject_id: str) -> TripDraft:
    return TripDraft(session_id=session_id, tenant_id=tenant_id, subject_id=subject_id)


__all__ = [
    "KNOWN_SLOTS",
    "OPTIONAL_SLOTS",
    "REQUIRED_SLOTS",
    "SLOT_HINTS",
    "SLOT_LABELS",
    "TripDraft",
    "new_trip_draft",
]
