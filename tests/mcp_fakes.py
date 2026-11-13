"""MCP 测试替身：一个只实现 `call/has/server_names` 的假 hub。

业务层（`LiveTravelData` 与工具）只依赖这三个成员，因此不必为单测起真实 MCP 服务。
替身按 `(server, tool)` 路由到处理器，并记录调用顺序，便于断言"站点代码先查、车次后查"。
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Mapping

from app.infrastructure.mcp.hub import BAZI_SERVER, FLIGHT_SERVER, GAODE_SERVER, TRAIN_SERVER, MCPError

Handler = Callable[[dict[str, Any]], Any]


class FakeMCPHub:
    def __init__(
        self,
        handlers: Mapping[tuple[str, str], Handler] | None = None,
        *,
        servers: Iterable[str] = (TRAIN_SERVER, GAODE_SERVER, BAZI_SERVER, FLIGHT_SERVER),
        fail_servers: Iterable[str] = (),
    ) -> None:
        self._handlers: dict[tuple[str, str], Handler] = dict(handlers or {})
        self._servers = tuple(servers)
        self._fail_servers = set(fail_servers)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    # ------------------------------------------------------------------ 契约
    def has(self, server: str) -> bool:
        return server in self._servers

    @property
    def server_names(self) -> tuple[str, ...]:
        return self._servers

    async def call(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        max_retries: int = 0,
        timeout_s: float | None = None,
    ) -> str:
        payload = dict(arguments or {})
        self.calls.append((server, tool, payload))
        if server in self._fail_servers:
            raise MCPError(f"{server} 不可用", server=server, tool=tool, retryable=True)
        handler = self._handlers.get((server, tool))
        if handler is None:
            raise MCPError(
                f"未预置 {server}.{tool} 的响应：{payload}",
                server=server,
                tool=tool,
                retryable=False,
            )
        result = handler(payload)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)

    # ------------------------------------------------------------------ 断言辅助
    def called(self, server: str, tool: str) -> list[dict[str, Any]]:
        return [args for s, t, args in self.calls if s == server and t == tool]

    def tools_called(self) -> list[str]:
        return [tool for _, tool, _ in self.calls]


# --------------------------------------------------------------------------- 常用响应


STATION_CODES = {
    "上海": "SHH",
    "杭州": "HZH",
    "北京": "BJP",
    "成都": "CDW",
    "苏州": "SZH",
}


def station_handler(args: dict[str, Any]) -> str:
    city = str(args.get("citys") or args.get("city") or args.get("cities") or "").split(",")[0]
    code = STATION_CODES.get(city)
    if code is None:
        return json.dumps({"error": f"未找到城市 {city}"}, ensure_ascii=False)
    return json.dumps(
        {city: [{"station_name": f"{city}站", "station_code": code}]}, ensure_ascii=False
    )


def tickets_handler(args: dict[str, Any]) -> str:
    return (
        f"车次查询结果（{args.get('fromStation')}→{args.get('toStation')}，{args.get('date')}）\n"
        "G7301 上海虹桥 07:00 → 杭州东 07:45 二等座 ¥73\n"
        "G7303 上海虹桥 08:00 → 杭州东 08:52 二等座 ¥73"
    )


def geo_handler(args: dict[str, Any]) -> dict[str, Any]:
    address = str(args.get("address") or args.get("city") or "")
    coords = {
        "上海": "121.473701,31.230416",
        "杭州": "120.155070,30.274084",
        "成都": "104.066541,30.572269",
    }.get(address, "120.000000,30.000000")
    return {"return": [{"location": coords, "formatted_address": address}]}


def driving_handler(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "route": {
            "paths": [
                {"distance": "176400", "duration": 8400, "tolls": "85", "toll_distance": "120000"},
                {"distance": "190000", "duration": 9000, "tolls": "95"},
            ]
        },
        "origin": args.get("origin"),
        "destination": args.get("destination"),
    }


def weather_handler(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "city": "杭州",
        "forecasts": [
            {
                "date": "2026-11-20",
                "dayweather": "晴",
                "nightweather": "多云",
                "daytemp": "20",
                "nighttemp": "12",
                "daywind": "东北风",
                "daypower": "3级",
            },
            {
                "date": "2026-11-21",
                "dayweather": "小雨",
                "nightweather": "阴",
                "daytemp": "17",
                "nighttemp": "11",
                "daywind": "北风",
            },
        ],
    }


def almanac_handler(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "公历": "2026-11-20",
        "农历": "十月初一",
        "干支": "丙午年 丁亥月 戊戌日",
        "宜": "出行,会友,开市",
        "忌": "动土,安葬",
    }


def hotel_handler(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "pois": [
            {"name": "甲酒店（西湖店）", "address": "北山街 1 号", "biz_ext": {"cost": "320"}},
            {"name": "乙酒店（钱江新城）", "address": "富春路 8 号", "biz_ext": {"cost": "980"}},
            {"name": "丙民宿", "address": "龙井路 12 号", "biz_ext": {"cost": "268"}},
        ],
        "count": "3",
    }


def flight_handler(args: dict[str, Any]) -> str:
    return (
        f"MU5401 {args.get('dep')}→{args.get('arr')} {args.get('date')} 08:00-11:05 空客A320 ¥980\n"
        f"CA4502 {args.get('dep')}→{args.get('arr')} {args.get('date')} 13:10-16:20 波音737 ¥1050"
    )


def default_handlers() -> dict[tuple[str, str], Handler]:
    return {
        (TRAIN_SERVER, "get-station-code-of-citys"): station_handler,
        (TRAIN_SERVER, "get-tickets"): tickets_handler,
        (GAODE_SERVER, "maps_geo"): geo_handler,
        (GAODE_SERVER, "maps_direction_driving"): driving_handler,
        (GAODE_SERVER, "maps_weather"): weather_handler,
        (GAODE_SERVER, "maps_text_search"): hotel_handler,
        (BAZI_SERVER, "getChineseCalendar"): almanac_handler,
        (FLIGHT_SERVER, "searchFlightsByDepArr"): flight_handler,
    }
