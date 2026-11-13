"""L2 槽位记忆的共享类型。

采购与差旅是两个业务域，但"槽位 = 值 + 来源 + 是否已确认"的结构是一样的：
模型推断出来的值不能覆盖用户已经确认的值。把这份结构抽出来，让两个域共用同一套
工具注册表与确认语义，而不是各写一份 `merge`。
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SlotSource(str, Enum):
    """槽位值的来源。可信度从高到低：主数据 > 用户 > 工具 > 模型推断。"""

    USER = "user"
    INFERRED = "inferred"
    MASTER_DATA = "master_data"
    TOOL = "tool"


class Slot(BaseModel):
    """单个槽位及其来源与确认状态。"""

    value: Any = None
    source: SlotSource = SlotSource.INFERRED
    confirmed: bool = False
    turn: int = 0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SlotChange(BaseModel):
    """被拒绝或被覆盖的槽位变更，供审计与提示模型。"""

    field: str
    old_value: Any = None
    new_value: Any = None
    reason: str


def to_jsonable(value: Any) -> Any:
    """把槽位值规范成 JSON 原生类型，保证可持久化且可稳定哈希。"""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = ["Slot", "SlotChange", "SlotSource", "to_jsonable"]
