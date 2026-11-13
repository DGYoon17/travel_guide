"""MCP 实时数据接入层。

- `hub.MCPToolHub`：服务器注册、传输选择（HTTP/SSE）、超时/重试/重建连接；
- `travel_data.LiveTravelData`：业务语义门面（车次/自驾/酒店/天气/黄历/航班）与宽容解析。

工具层（`app/core/tools/travel_mcp.py`）只依赖这两层，不直接碰 JSON-RPC。
"""

from __future__ import annotations

from app.infrastructure.mcp.hub import (
    BAZI_SERVER,
    FLIGHT_SERVER,
    GAODE_SERVER,
    TRAIN_SERVER,
    MCPError,
    MCPServerConfig,
    MCPToolHub,
    build_mcp_hub,
    parse_server_configs,
    resolve_server_configs,
)
from app.infrastructure.mcp.travel_data import (
    AirportCodeError,
    Almanac,
    DrivingRoute,
    FlightSearch,
    HotelHit,
    HotelSearch,
    LiveTravelData,
    StationCode,
    TrainTickets,
    WeatherDay,
    WeatherReport,
)

__all__ = [
    "BAZI_SERVER",
    "FLIGHT_SERVER",
    "GAODE_SERVER",
    "TRAIN_SERVER",
    "AirportCodeError",
    "Almanac",
    "DrivingRoute",
    "FlightSearch",
    "HotelHit",
    "HotelSearch",
    "LiveTravelData",
    "MCPError",
    "MCPServerConfig",
    "MCPToolHub",
    "StationCode",
    "TrainTickets",
    "WeatherDay",
    "WeatherReport",
    "build_mcp_hub",
    "parse_server_configs",
    "resolve_server_configs",
]
