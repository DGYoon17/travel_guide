"""参数宽松解析的测试。

覆盖改造前会直接抛 `ValueError`/`KeyError` 冒泡成 500 的输入：模型把 `needed_by` 填成
「下周五」、把金额填成「1.5万」。
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.core.tools.coercion import parse_cn_date, parse_cny, parse_positive_int
from app.core.tools.errors import ToolError, ToolErrorCode

# 2026-08-20 是星期四，便于验证 本周/下周 的计算。
REF = date(2026, 8, 20)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-09-15", date(2026, 9, 15)),
        ("2026/09/15", date(2026, 9, 15)),
        ("2026年9月15日", date(2026, 9, 15)),
        ("20260915", date(2026, 9, 15)),
        ("今天", REF),
        ("明天", date(2026, 8, 21)),
        ("后天", date(2026, 8, 22)),
        ("7天后", date(2026, 8, 27)),
        ("2周后", date(2026, 9, 3)),
        ("1个月后", date(2026, 9, 20)),
        ("本月底", date(2026, 8, 31)),
        ("下月初", date(2026, 9, 1)),
    ],
)
def test_parse_absolute_and_relative_dates(raw: str, expected: date) -> None:
    assert parse_cn_date(raw, field="needed_by", today=REF) == expected


def test_parse_next_week_friday() -> None:
    """「下周五」是改造前直接崩掉的典型输入（int("下周五") → ValueError）。"""
    assert parse_cn_date("下周五", field="needed_by", today=REF) == date(2026, 8, 28)


def test_parse_this_week_and_bare_weekday() -> None:
    assert parse_cn_date("本周五", field="needed_by", today=REF) == date(2026, 8, 21)
    # 无前缀的「周三」取最近的将来那一天（参考日是周四，所以落到下周三）。
    assert parse_cn_date("周三", field="needed_by", today=REF) == date(2026, 8, 26)


def test_business_days_skip_weekend() -> None:
    # 周四 + 3 个工作日 = 下周二（跳过周六周日）。
    assert parse_cn_date("3个工作日后", field="needed_by", today=REF) == date(2026, 8, 25)


def test_month_day_rolls_to_next_year_when_past() -> None:
    """只给月日且已过去时应落到明年，而不是把需求日期解析到过去。"""
    assert parse_cn_date("1月5日", field="needed_by", today=REF) == date(2027, 1, 5)
    assert parse_cn_date("12月5日", field="needed_by", today=REF) == date(2026, 12, 5)


def test_unparseable_date_raises_tool_error_with_guidance() -> None:
    with pytest.raises(ToolError) as excinfo:
        parse_cn_date("等我问一下老板", field="needed_by", today=REF)
    err = excinfo.value
    assert err.code is ToolErrorCode.INVALID_ARGUMENTS
    assert err.repairable is True
    assert err.field_issues[0].field == "needed_by"
    # 关键：错误里要带期望格式和示例，模型才能自修正。
    assert err.field_issues[0].expected
    assert err.field_issues[0].example == "2026-09-15"


def test_invalid_calendar_date_raises() -> None:
    with pytest.raises(ToolError):
        parse_cn_date("2026-02-30", field="needed_by", today=REF)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (15000, Decimal("15000.00")),
        ("15000", Decimal("15000.00")),
        ("15,000元", Decimal("15000.00")),
        ("￥15000", Decimal("15000.00")),
        ("1.5万", Decimal("15000.00")),
        ("1万5", Decimal("15000.00")),
        ("8千", Decimal("8000.00")),
        ("18.8k", Decimal("18800.00")),
        ("人民币 6299 元", Decimal("6299.00")),
    ],
)
def test_parse_money(raw: object, expected: Decimal) -> None:
    assert parse_cny(raw, field="unit_price_cny") == expected


def test_negative_and_zero_money() -> None:
    with pytest.raises(ToolError):
        parse_cny("-100", field="unit_price_cny")
    with pytest.raises(ToolError):
        parse_cny("0", field="unit_price_cny", allow_zero=False)
    assert parse_cny("0", field="budget_ceiling_cny", allow_zero=True) == Decimal("0.00")


def test_money_error_is_repairable_with_example() -> None:
    with pytest.raises(ToolError) as excinfo:
        parse_cny("看情况", field="unit_price_cny")
    issue = excinfo.value.field_issues[0]
    assert issue.field == "unit_price_cny"
    assert issue.example == "15000"


@pytest.mark.parametrize(("raw", "expected"), [(3, 3), ("3", 3), ("3个", 3), (3.0, 3)])
def test_parse_quantity(raw: object, expected: int) -> None:
    assert parse_positive_int(raw, field="quantity") == expected


def test_parse_quantity_rejects_fraction_and_out_of_range() -> None:
    with pytest.raises(ToolError):
        parse_positive_int(2.5, field="quantity")
    with pytest.raises(ToolError):
        parse_positive_int(0, field="quantity")
    with pytest.raises(ToolError):
        parse_positive_int(100000, field="quantity")
    with pytest.raises(ToolError):
        parse_positive_int(True, field="quantity")
