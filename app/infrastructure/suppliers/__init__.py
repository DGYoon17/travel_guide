"""差旅供应方适配层。

真实项目里这里对接 TMC / 航司 / 酒店 API。本仓库用确定性 mock 实现，目的是让
"搜索 → 报价快照 → 确认 → 二次校验 → 预订 → 退订"整条链路可以离线跑通、可以单测。

**不要把 mock 说成已接入**。面试时正确的说法是：通过 `TravelSupplier` 协议抽象，
当前是 mock 实现，替换实现即可接 TMC。
"""

from __future__ import annotations

from app.infrastructure.suppliers.mock_travel import (
    MockTravelSupplier,
    SupplierError,
    TravelSupplier,
)

__all__ = ["MockTravelSupplier", "SupplierError", "TravelSupplier"]
