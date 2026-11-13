"""业务时区与当前时间的统一入口。

两个原因需要它：

1. **容器时区**。服务通常跑在 UTC，而采购的"需求日期""提前几天发起"都是按国内日历算的。
   直接用 `date.today()` 会在 UTC 16:00 之后整体错开一天，进而误判"是否低于最小提前天数"。
2. **tzdata 缺失**。`ZoneInfo("Asia/Shanghai")` 在未安装 tzdata 的环境（典型是 Windows、
   精简版容器镜像）会抛 `ZoneInfoNotFoundError`。改造前 `requisition.py` 在函数体内直接构造
   `ZoneInfo`，这个错误要等到真的生成请购单时才炸。这里改为启动即解析，失败则回落到固定
   +08:00 偏移 —— 对本场景语义等价（中国大陆无夏令时）。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone, tzinfo

_TZ_NAME = "Asia/Shanghai"
_FIXED_OFFSET = timezone(timedelta(hours=8), name="UTC+08:00")


def _resolve_business_tz() -> tzinfo:
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(_TZ_NAME)
    except Exception:  # noqa: BLE001 - ZoneInfoNotFoundError 及其它导入期异常
        return _FIXED_OFFSET


#: 业务时区。已安装 tzdata 时为 Asia/Shanghai，否则为固定 +08:00。
BUSINESS_TZ: tzinfo = _resolve_business_tz()


def business_now() -> datetime:
    return datetime.now(BUSINESS_TZ)


def business_today() -> date:
    return business_now().date()


def at_business_time(day: date, hour: int = 9, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=BUSINESS_TZ)
