"""参数宽松解析：把模型/用户的自然语言表达转成严格类型。

原来的实现是 `date(*map(int, s.split("-")))` 和 `Decimal(str(x))`，模型填「下周五」或
「1.5万」就直接抛 `ValueError` 冒泡成 500。这里做两件事：

1. 能正确解析的，解析掉（大部分情况）——这才是真正减少失败轮次的办法；
2. 确实解析不了的，抛 `ToolError` 并带上期望格式与示例，让模型下一轮自修正。
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta
from decimal import Decimal, DecimalException, InvalidOperation
from typing import Any

from app.core.clock import BUSINESS_TZ, business_today
from app.core.tools.errors import FieldIssue, ToolError, ToolErrorCode

__all__ = [
    "BUSINESS_TZ",
    "business_today",
    "parse_cn_date",
    "parse_cny",
    "parse_positive_int",
]

_WEEKDAY_CN: dict[str, int] = {
    "一": 0, "1": 0,
    "二": 1, "2": 1,
    "三": 2, "3": 2,
    "四": 3, "4": 3,
    "五": 4, "5": 4,
    "六": 5, "6": 5,
    "日": 6, "天": 6, "7": 6,
}

_ISO_RE = re.compile(r"^(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})\s*日?$")
_COMPACT_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_MONTH_DAY_RE = re.compile(r"^(\d{1,2})\s*[-/.月]\s*(\d{1,2})\s*日?$")
_REL_DAYS_RE = re.compile(r"^(\d+)\s*(?:个)?\s*(工作日|天|日)\s*(?:之?[后内以]?内?|之?后)?$")
_REL_WEEKS_RE = re.compile(r"^(\d+)\s*(?:个)?\s*(?:周|星期)\s*(?:之?[后内]|之?后)?$")
_REL_MONTHS_RE = re.compile(r"^(\d+)\s*个?\s*月\s*(?:之?[后内]|之?后)?$")
_WEEKDAY_RE = re.compile(
    r"^(?P<prefix>本|这|下下|下|上)?\s*(?:个)?\s*(?:周|星期|礼拜)\s*(?P<day>[一二三四五六日天1-7])$"
)
_DAY_OF_MONTH_RE = re.compile(r"^(?P<prefix>本|这|下)\s*个?月\s*(?P<day>\d{1,2})\s*[日号]$")

_FIXED_OFFSETS: dict[str, int] = {
    "今天": 0, "今日": 0, "当天": 0, "今儿": 0,
    "明天": 1, "明日": 1, "次日": 1,
    "后天": 2, "后日": 2,
    "大后天": 3,
}

_DATE_EXPECTED = "ISO 日期 YYYY-MM-DD；也接受「明天」「下周五」「7天后」「本月底」等相对表述"
_DATE_EXAMPLE = "2026-09-15"


def _add_business_days(start: date, days: int) -> date:
    cur = start
    remaining = days
    while remaining > 0:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            remaining -= 1
    return cur


def _add_months(start: date, months: int) -> date:
    total = start.month - 1 + months
    year = start.year + total // 12
    month = total % 12 + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _weekday_target(today: date, weekday: int, prefix: str | None) -> date:
    """本周/下周/下下周的某个工作日；无前缀时取最近的将来那一天。"""
    monday = today - timedelta(days=today.weekday())
    if prefix in ("本", "这"):
        return monday + timedelta(days=weekday)
    if prefix == "下":
        return monday + timedelta(days=7 + weekday)
    if prefix == "下下":
        return monday + timedelta(days=14 + weekday)
    if prefix == "上":
        return monday - timedelta(days=7 - weekday)
    delta = (weekday - today.weekday()) % 7
    return today + timedelta(days=delta or 7)


def _date_error(field: str, raw: Any) -> ToolError:
    return ToolError(
        ToolErrorCode.INVALID_ARGUMENTS,
        f"字段 {field} 的日期「{raw}」无法解析",
        field_issues=[
            FieldIssue(field=field, problem="日期格式无法识别", expected=_DATE_EXPECTED, example=_DATE_EXAMPLE)
        ],
        hint="若用户未给出明确日期，请直接询问，不要猜测。",
    )


def parse_cn_date(value: Any, *, field: str = "date", today: date | None = None) -> date:
    """把多种中英文日期表达解析为 `date`。无法解析时抛 `ToolError`。"""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if value is None:
        raise _date_error(field, value)

    ref = today or business_today()
    text = str(value).strip().replace(" ", "").replace("　", "")
    if not text:
        raise _date_error(field, value)

    if text in _FIXED_OFFSETS:
        return ref + timedelta(days=_FIXED_OFFSETS[text])
    if text in ("本月底", "月底", "这个月底"):
        return date(ref.year, ref.month, calendar.monthrange(ref.year, ref.month)[1])
    if text in ("下月初", "下个月初"):
        return _add_months(date(ref.year, ref.month, 1), 1)
    if text in ("下月底", "下个月底"):
        nxt = _add_months(date(ref.year, ref.month, 1), 1)
        return date(nxt.year, nxt.month, calendar.monthrange(nxt.year, nxt.month)[1])

    if m := _ISO_RE.match(text):
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)), field, value)
    if m := _COMPACT_RE.match(text):
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)), field, value)
    if m := _MONTH_DAY_RE.match(text):
        month, day = int(m.group(1)), int(m.group(2))
        candidate = _safe_date(ref.year, month, day, field, value)
        # 只写了月日时，取不早于今天的那一年，避免把需求日期解析到过去。
        return candidate if candidate >= ref else _safe_date(ref.year + 1, month, day, field, value)
    if m := _REL_DAYS_RE.match(text):
        n = int(m.group(1))
        return _add_business_days(ref, n) if m.group(2) == "工作日" else ref + timedelta(days=n)
    if m := _REL_WEEKS_RE.match(text):
        return ref + timedelta(weeks=int(m.group(1)))
    if m := _REL_MONTHS_RE.match(text):
        return _add_months(ref, int(m.group(1)))
    if m := _WEEKDAY_RE.match(text):
        return _weekday_target(ref, _WEEKDAY_CN[m.group("day")], m.group("prefix"))
    if m := _DAY_OF_MONTH_RE.match(text):
        base = ref if m.group("prefix") in ("本", "这") else _add_months(ref, 1)
        day = min(int(m.group("day")), calendar.monthrange(base.year, base.month)[1])
        return date(base.year, base.month, day)

    raise _date_error(field, value)


def _safe_date(year: int, month: int, day: int, field: str, raw: Any) -> date:
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"字段 {field} 的日期「{raw}」不是合法日历日期：{exc}",
            field_issues=[
                FieldIssue(field=field, problem=str(exc), expected=_DATE_EXPECTED, example=_DATE_EXAMPLE)
            ],
        ) from exc


_MONEY_UNITS: tuple[tuple[str, Decimal], ...] = (
    ("亿", Decimal("100000000")),
    ("万", Decimal("10000")),
    ("w", Decimal("10000")),
    ("千", Decimal("1000")),
    ("k", Decimal("1000")),
    ("百", Decimal("100")),
)

_MONEY_NOISE = re.compile(r"(人民币|rmb|cny|¥|￥|元整|元|块钱|块|,|，|\s)", re.IGNORECASE)
_COMPOSITE_MONEY_RE = re.compile(r"^(\d+(?:\.\d+)?)万(\d+(?:\.\d+)?)(?:千)?$")

_MONEY_EXPECTED = "以元为单位的数字；也接受「1.5万」「8千」「15,000元」"
_MONEY_EXAMPLE = "15000"


def _money_error(field: str, raw: Any, problem: str) -> ToolError:
    return ToolError(
        ToolErrorCode.INVALID_ARGUMENTS,
        f"字段 {field} 的金额「{raw}」无法解析：{problem}",
        field_issues=[
            FieldIssue(field=field, problem=problem, expected=_MONEY_EXPECTED, example=_MONEY_EXAMPLE)
        ],
        hint="金额必须来自用户明确给出的数字，不要估算。",
    )


def parse_cny(value: Any, *, field: str = "amount_cny", allow_zero: bool = True) -> Decimal:
    """把金额表达解析为以元为单位的 `Decimal`。"""
    if value is None:
        raise _money_error(field, value, "值为空")
    if isinstance(value, bool):
        raise _money_error(field, value, "布尔值不是金额")
    if isinstance(value, (int, float, Decimal)):
        amount = Decimal(str(value))
        return _validate_money(amount, field, value, allow_zero)

    text = _MONEY_NOISE.sub("", str(value).strip()).lower()
    if not text:
        raise _money_error(field, value, "去除单位后为空")

    if m := _COMPOSITE_MONEY_RE.match(text):
        # 「1万5」= 15000；「1万5千」同义。
        head, tail = Decimal(m.group(1)), Decimal(m.group(2))
        amount = head * Decimal("10000") + tail * Decimal("1000")
        return _validate_money(amount, field, value, allow_zero)

    multiplier = Decimal("1")
    for suffix, factor in _MONEY_UNITS:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            multiplier = factor
            break

    if not text:
        raise _money_error(field, value, "只有单位没有数字")
    try:
        amount = Decimal(text) * multiplier
    except (InvalidOperation, DecimalException, ValueError) as exc:
        raise _money_error(field, value, "不是有效数字") from exc
    return _validate_money(amount, field, value, allow_zero)


def _validate_money(amount: Decimal, field: str, raw: Any, allow_zero: bool) -> Decimal:
    if amount.is_nan() or amount.is_infinite():
        raise _money_error(field, raw, "不是有限数值")
    if amount < 0:
        raise _money_error(field, raw, "金额不能为负")
    if not allow_zero and amount == 0:
        raise _money_error(field, raw, "金额必须大于 0")
    return amount.quantize(Decimal("0.01"))


def parse_positive_int(value: Any, *, field: str, maximum: int = 9999) -> int:
    """数量类字段：接受 "3"、"3个"、3.0（整数值）。"""
    if isinstance(value, bool):
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"字段 {field} 需要整数，收到布尔值",
            field_issues=[FieldIssue(field=field, problem="类型错误", expected="1 到 9999 的整数", example="3")],
        )
    raw = value
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.]", "", value.strip())
        value = cleaned or None
    if value is None:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"字段 {field} 无法解析为整数：{raw!r}",
            field_issues=[FieldIssue(field=field, problem="无法解析", expected="1 到 9999 的整数", example="3")],
        )
    try:
        number = Decimal(str(value))
    except (InvalidOperation, DecimalException, ValueError) as exc:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"字段 {field} 无法解析为整数：{raw!r}",
            field_issues=[FieldIssue(field=field, problem="无法解析", expected="1 到 9999 的整数", example="3")],
        ) from exc
    if number != number.to_integral_value():
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"字段 {field} 必须是整数，收到 {raw!r}",
            field_issues=[FieldIssue(field=field, problem="含小数", expected="1 到 9999 的整数", example="3")],
        )
    result = int(number)
    if not 1 <= result <= maximum:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"字段 {field} 超出允许范围 1~{maximum}：{result}",
            field_issues=[
                FieldIssue(field=field, problem="超出范围", expected=f"1 到 {maximum} 的整数", example="3")
            ],
        )
    return result
