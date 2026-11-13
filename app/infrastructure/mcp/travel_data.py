"""实时差旅数据适配层：把 MCP 返回的"任意形状"归一化成可渲染、可断言的结构。

真实 MCP 服务的返回形态很不统一：高德官方字段（`distance` 米、`duration` 秒）、
包装过的字段（`distance_km`、`duration_minutes`）、以及直接返回一段中文文本。
这一层统一处理三件事：

1. **宽容解析**：JSON 优先；拿不到 JSON 就退化为文本解析（酒店名、站点代码）；
2. **明确降级**：解析不出关键字段时**不猜数值**，而是标注"无法解析"，让模型如实说明；
3. **可测试**：解析函数都是纯函数，工具层只负责渲染与错误翻译。

距离在毫秒必争的场景下只做估算（高德不可用时按城市中心点 ×1.25 路网系数），
且必须带 `estimated=True` 标记——降级可以，伪装成实时数据不行。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Iterator, Mapping, Sequence

from app.domain.travel.geo import (
    CITY_COORDS,
    estimate_distance,
    lookup_airport,
    normalize_city,
)
from app.infrastructure.mcp.hub import (
    BAZI_SERVER,
    FLIGHT_SERVER,
    GAODE_SERVER,
    TRAIN_SERVER,
    MCPError,
)

#: 单次工具返回文本的上限。车次列表可能很长，但上下文不是无限的。
MAX_TEXT_CHARS = 12_000

#: 站点代码查询很少变，缓存一天；失败不缓存（下次还会重试）。
DEFAULT_STATION_TTL_S = 24 * 3600


class AirportCodeError(ValueError):
    """城市名无法转换为机场三字码。工具层据此返回字段级错误。"""

    def __init__(self, city: str) -> None:
        super().__init__(f"无法把「{city}」转换为机场三字码")
        self.city = city


# --------------------------------------------------------------------------- 结构


@dataclass(frozen=True)
class StationCode:
    city: str
    code: str
    name: str = ""


@dataclass
class TrainTickets:
    origin: str
    destination: str
    date: str
    text: str
    from_code: str | None = None
    to_code: str | None = None
    stations_resolved: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        route = f"{self.origin}→{self.destination}"
        suffix = ""
        if self.from_code and self.to_code:
            suffix = f"（{self.from_code}→{self.to_code}）"
        return f"{route}{suffix} {self.date}"


@dataclass
class RouteFacts:
    distance_km: float | None = None
    duration_minutes: int | None = None
    tolls_cny: float | None = None
    toll_distance_km: float | None = None


@dataclass
class DrivingRoute:
    origin: str
    destination: str
    distance_km: float | None = None
    duration_minutes: int | None = None
    tolls_cny: float | None = None
    toll_distance_km: float | None = None
    estimated: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class HotelHit:
    name: str
    address: str = ""
    cost_cny: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class HotelSearch:
    city: str
    keywords: list[str]
    hits: list[HotelHit]
    budget_per_night: float | None = None
    over_budget: int = 0
    structured: bool = False
    text: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class WeatherDay:
    date: str
    day_weather: str = ""
    night_weather: str = ""
    day_temp: str = ""
    night_temp: str = ""
    wind: str = ""


@dataclass
class WeatherReport:
    city: str
    days: list[WeatherDay] = field(default_factory=list)
    available_from: str = ""
    available_to: str = ""
    missing_dates: list[str] = field(default_factory=list)
    structured: bool = False
    text: str = ""


@dataclass
class Almanac:
    solar_date: str
    lunar_date: str = ""
    ganzhi: str = ""
    yi: str = ""
    ji: str = ""
    auspicious: bool | None = None
    text: str = ""


@dataclass
class FlightSearch:
    dep_city: str
    arr_city: str
    dep_code: str
    arr_code: str
    date: str
    text: str


# --------------------------------------------------------------------------- 通用解析


def _loads(raw: Any) -> Any | None:
    """尽力把返回值解析成 JSON；解析不了返回 None（调用方走文本分支）。"""
    if isinstance(raw, (dict, list)):
        return raw
    text = str(raw or "").strip()
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _iter_dicts(payload: Any, *, max_nodes: int = 400) -> Iterator[dict[str, Any]]:
    """广度优先遍历嵌套结构里的所有 dict。用于"字段藏在第几层都不怕"的解析。"""
    queue: list[Any] = [payload]
    visited = 0
    while queue and visited < max_nodes:
        node = queue.pop(0)
        visited += 1
        if isinstance(node, Mapping):
            yield dict(node)
            queue.extend(node.values())
        elif isinstance(node, list):
            queue.extend(node[:20])


def _iter_lists(payload: Any, keys: Sequence[str], *, max_nodes: int = 400) -> list[list[Any]]:
    """收集所有位于 `keys` 下的列表。"""
    wanted = set(keys)
    found: list[list[Any]] = []
    for node in _iter_dicts(payload, max_nodes=max_nodes):
        for key in wanted:
            value = node.get(key)
            if isinstance(value, list) and value:
                found.append(value)
    if isinstance(payload, list) and payload:
        found.append(payload)
    return found


def _to_float(value: Any) -> float | None:
    """从数字/字符串里取出一个数字（忽略单位文本）。无法解析返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("，", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:  # pragma: no cover - 正则已保证可解析
        return None


_DURATION_TOKEN_RE = re.compile(r"(?:(\d+(?:\.\d+)?)\s*(?:小时|时|h|hour))?\s*(?:(\d+(?:\.\d+)?)\s*(?:分钟|分|min|m))?")


def _duration_minutes_from_text(text: str) -> float | None:
    match = _DURATION_TOKEN_RE.search(text.strip())
    if match is None:
        return None
    hours, minutes = match.group(1), match.group(2)
    if hours is None and minutes is None:
        return None
    total = 0.0
    if hours is not None:
        total += float(hours) * 60
    if minutes is not None:
        total += float(minutes)
    return total


def _distance_km(value: Any) -> float | None:
    """距离 -> 公里。高德原始字段是米；带 km/公里 单位时按公里解释。"""
    if value is None:
        return None
    text = str(value).lower()
    number = _to_float(value)
    if number is None:
        return None
    if "km" in text or "公里" in text:
        return round(number, 2)
    return round(number / 1000.0, 2)


def _duration_minutes(value: Any) -> int | None:
    """时长 -> 分钟。高德原始字段是秒；带单位文本时按文本解释。"""
    if value is None:
        return None
    text = str(value).lower()
    if any(unit in text for unit in ("小时", "时", "h", "min", "分钟", "分")) and not text.replace(
        ".", ""
    ).isdigit():
        minutes = _duration_minutes_from_text(text)
        return int(round(minutes)) if minutes is not None else None
    number = _to_float(value)
    if number is None:
        return None
    return int(round(number / 60.0))


def _tolls_cny(value: Any) -> float | None:
    if value is None:
        return None
    number = _to_float(value)
    return round(number, 2) if number is not None else None


def _coords_from(value: Any) -> str | None:
    """把多种坐标表示归一化成高德要的 `"经度,纬度"`。"""
    if value is None:
        return None
    lng: Any = None
    lat: Any = None
    if isinstance(value, str):
        parts = re.split(r"[,，\s]+", value.strip())
        if len(parts) >= 2:
            lng, lat = parts[0], parts[1]
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        lng, lat = value[0], value[1]
    elif isinstance(value, Mapping):
        lng = value.get("lng") or value.get("lon") or value.get("longitude")
        lat = value.get("lat") or value.get("latitude")
    if lng is None or lat is None:
        return None
    try:
        lng_f, lat_f = float(lng), float(lat)
    except (TypeError, ValueError):
        return None
    if not (-180.0 <= lng_f <= 180.0 and -90.0 <= lat_f <= 90.0):
        return None
    if lng_f == 0.0 and lat_f == 0.0:
        return None
    return f"{lng_f:.6f},{lat_f:.6f}"


def parse_geo_payload(payload: Any) -> str | None:
    """从地理编码返回中取第一个可信坐标。"""
    if payload is None:
        return None
    if isinstance(payload, str):
        return _coords_from(payload)
    for node in _iter_dicts(payload):
        for key in ("location", "center", "coordinates"):
            coords = _coords_from(node.get(key))
            if coords:
                return coords
        lng = node.get("lng") or node.get("lon") or node.get("longitude")
        lat = node.get("lat") or node.get("latitude")
        if lng is not None and lat is not None:
            coords = _coords_from([lng, lat])
            if coords:
                return coords
    return None


# --------------------------------------------------------------------------- 站点代码

_STATION_CODE_KEYS = ("station_code", "stationCode", "telecode", "code", "abbr", "三字码")
_STATION_NAME_KEYS = ("station_name", "stationName", "name", "station", "站名")


def _looks_like_station(node: Mapping[str, Any]) -> bool:
    return any(isinstance(node.get(k), str) and str(node.get(k)).strip() for k in _STATION_CODE_KEYS)


def _station_from(node: Mapping[str, Any], city: str) -> StationCode | None:
    code = ""
    for key in _STATION_CODE_KEYS:
        value = node.get(key)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z]{2,6}", value.strip()):
            code = value.strip().upper()
            break
    if not code:
        return None
    name = ""
    for key in _STATION_NAME_KEYS:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            name = value.strip()
            break
    return StationCode(city=city, code=code, name=name)


def parse_station_payload(payload: Any, city: str) -> list[StationCode]:
    """解析 12306 站点代码返回。

    兼容四种已知形态：
    - `{"上海": [{"station_code": "SHH", "station_name": "上海"}]}`（按城市分组）
    - `{"data": [{"city": "上海", "stations": [...]}]}`
    - `[{"station_name": "上海", "station_code": "SHH"}, ...]`
    - 以上形态的任意嵌套包装（`return` / `result` / `data`）
    """
    text = normalize_city(city)
    collected: list[StationCode] = []
    nodes = list(_iter_dicts(payload))

    # 1) 按城市分组：key 命中城市名时直接取该分组
    for node in nodes:
        for key, value in node.items():
            if not isinstance(key, str) or normalize_city(key) != text:
                continue
            entries = value if isinstance(value, list) else [value]
            for entry in entries:
                if isinstance(entry, Mapping):
                    station = _station_from(entry, text)
                    if station:
                        collected.append(station)

    # 2) 带归属城市的嵌套结构：按父级 city 过滤，取该城市的车站
    if not collected:

        def walk(node: Any, inherited: str | None) -> None:
            if isinstance(node, Mapping):
                declared = (
                    node.get("city") or node.get("city_name") or node.get("cityName")
                )
                current = normalize_city(str(declared)) if declared else inherited
                if declared and current != text:
                    # 整个子树属于别的城市，不进去避免"杭州的车站混进上海"。
                    return
                if _looks_like_station(node) and (current is None or current == text):
                    station = _station_from(node, text)
                    if station:
                        collected.append(station)
                for value in node.values():
                    walk(value, current)
            elif isinstance(node, list):
                for item in node:
                    walk(item, inherited)

        walk(payload, None)

    # 3) 纯文本兜底：`上海: SHH` / `上海站 SHH`
    if not collected and isinstance(payload, str):
        collected = parse_stations_from_text(payload, text)

    seen: set[str] = set()
    unique: list[StationCode] = []
    for station in collected:
        if station.code in seen:
            continue
        seen.add(station.code)
        unique.append(station)
    return unique


def parse_stations_from_text(text: str, city: str) -> list[StationCode]:
    """从 `上海: SHH` 这类文本里提取站点三字码。只在 JSON 解析失败时使用。"""
    target = normalize_city(city)
    pattern = re.compile(r"([\u4e00-\u9fa5]{2,10})\s*(?:站)?\s*(?:[:：\-—|]|\s)\s*([A-Z]{3})\b")
    results: list[StationCode] = []
    for name, code in pattern.findall(str(text or "")):
        results.append(StationCode(city=target, code=code, name=name))
    if not results:
        return []
    # 城市名完全匹配的排前面，其余作为候选（例如「上海虹桥」之于「上海」）。
    results.sort(key=lambda s: 0 if normalize_city(s.name) == target else 1)
    return results


# --------------------------------------------------------------------------- 路线


def _route_facts_from(node: Mapping[str, Any]) -> RouteFacts:
    # 字段名自带单位时不要再除以 1000，否则 123km 会被读成 0.12km。
    distance: float | None = None
    for key in ("distance_km", "distanceKm", "km"):
        if node.get(key) is not None:
            raw = _to_float(node.get(key))
            distance = round(raw, 2) if raw is not None else None
            break
    if distance is None:
        distance = _distance_km(
            node.get("distance")
            if node.get("distance") is not None
            else node.get("distance_m", node.get("distanceMeters"))
        )
    duration = None
    for key in ("duration_minutes", "durationMinutes"):
        if node.get(key) is not None:
            raw = _to_float(node.get(key))
            duration = int(round(raw)) if raw is not None else None
            break
    if duration is None:
        duration = _duration_minutes(
            node.get("duration")
            if node.get("duration") is not None
            else node.get("duration_s", node.get("durationSeconds", node.get("eta")))
        )
    tolls = None
    for key in ("tolls", "toll", "toll_cost", "tolls_cny", "tollCny"):
        if node.get(key) is not None:
            tolls = _tolls_cny(node.get(key))
            break
    toll_distance = None
    for key in ("toll_distance", "tollDistance", "toll_km"):
        if node.get(key) is not None:
            toll_distance = (
                _distance_km(node.get(key))
                if key != "toll_km"
                else round(_to_float(node.get(key)) or 0.0, 2)
            )
            break
    return RouteFacts(
        distance_km=distance if distance is not None and distance > 0 else None,
        duration_minutes=duration,
        tolls_cny=tolls,
        toll_distance_km=toll_distance,
    )


def parse_route_payload(payload: Any) -> RouteFacts:
    """解析驾车路线。

    `paths` 在高德语义里是**备选路线**，因此取最短的那条而不是求和；
    直接把一组 dict 当顶层返回的服务（少见）按同一规则处理，避免里程翻倍。
    """
    if payload is None:
        return RouteFacts()
    if isinstance(payload, str):
        return RouteFacts()

    path_lists = _iter_lists(payload, ("paths", "routes", "route"))
    for paths in path_lists:
        candidates = [
            item
            for item in paths
            if isinstance(item, Mapping) and item.get("distance") is not None
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda item: _to_float(item.get("distance")) or float("inf"))
        facts = _route_facts_from(best)
        if facts.distance_km is not None:
            return facts

    for node in _iter_dicts(payload):
        if any(
            node.get(key) is not None
            for key in ("distance", "distance_km", "distanceMeters", "duration")
        ):
            facts = _route_facts_from(node)
            if facts.distance_km is not None:
                return facts
    return RouteFacts()


# --------------------------------------------------------------------------- POI / 酒店

_POI_NAME_KEYS = ("name", "title", "poi_name", "poiName")
_POI_ADDRESS_KEYS = ("address", "addr", "location", "address_name")
_POI_COST_PATHS: tuple[tuple[str, ...], ...] = (
    ("biz_ext", "cost"),
    ("biz_ext", "price"),
    ("biz_ext", "lowest_price"),
    ("cost",),
    ("price",),
    ("min_cost",),
    ("avg_price",),
    ("reference_price",),
)


def _poi_cost(node: Mapping[str, Any]) -> float | None:
    for path in _POI_COST_PATHS:
        cursor: Any = node
        for key in path:
            if not isinstance(cursor, Mapping) or key not in cursor:
                cursor = None
                break
            cursor = cursor[key]
        if cursor is None:
            continue
        value = _to_float(cursor)
        if value is not None and value > 0:
            return value
    return None


def _poi_from(node: Mapping[str, Any]) -> HotelHit | None:
    name = ""
    for key in _POI_NAME_KEYS:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            name = value.strip()
            break
    if not name:
        return None
    address = ""
    for key in _POI_ADDRESS_KEYS:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            address = value.strip()
            break
    return HotelHit(
        name=name,
        address=address,
        cost_cny=_poi_cost(node),
        raw={k: v for k, v in node.items() if isinstance(v, (str, int, float))},
    )


_TEXT_POI_RE = re.compile(r"^\s*(?:[-*]|\d+[.、)])\s*(.+?)\s*$")
_TEXT_COST_RE = re.compile(r"(?:¥|￥|人均|价格|参考价|起价)\s*[:：]?\s*(\d+(?:\.\d+)?)")


def parse_pois_from_text(text: str) -> list[HotelHit]:
    """从 markdown/编号文本里提取 POI 名称与价格。信息有限，只做尽力而为。"""
    hits: list[HotelHit] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        match = _TEXT_POI_RE.match(line)
        if match is None:
            continue
        name = match.group(1).strip().strip("*").strip()
        name = re.sub(r"^「|」$", "", name)
        # 同一条目里常带"人均¥280""价格：￥1200"这类后缀，先摘掉再当酒店名。
        name = re.split(r"(?:¥|￥|人均|价格|参考价|起价)", name)[0].strip(" -—|｜,，:：")
        if len(name) < 2 or name.startswith(("地址", "电话", "评分", "距离")):
            continue
        cost_match = _TEXT_COST_RE.search(line)
        hits.append(
            HotelHit(
                name=name,
                cost_cny=float(cost_match.group(1)) if cost_match else None,
            )
        )
    # 去重保序
    seen: set[str] = set()
    unique: list[HotelHit] = []
    for hit in hits:
        if hit.name in seen:
            continue
        seen.add(hit.name)
        unique.append(hit)
    return unique


def parse_poi_payload(payload: Any) -> list[HotelHit]:
    """解析 POI / 酒店搜索结果。JSON 优先，失败时返回空列表由调用方走文本渲染。"""
    if payload is None:
        return []
    if isinstance(payload, str):
        return parse_pois_from_text(payload)
    hits: list[HotelHit] = []
    for node in _iter_dicts(payload):
        if not _looks_like_poi(node):
            continue
        hit = _poi_from(node)
        if hit is not None:
            hits.append(hit)
    seen: set[str] = set()
    unique: list[HotelHit] = []
    for hit in hits:
        if hit.name in seen:
            continue
        seen.add(hit.name)
        unique.append(hit)
    return unique


def _looks_like_poi(node: Mapping[str, Any]) -> bool:
    if not any(isinstance(node.get(k), str) and node.get(k) for k in _POI_NAME_KEYS):
        return False
    return any(
        key in node
        for key in ("address", "location", "biz_ext", "type", "poi_id", "id", "distance")
    ) or any(key in node for key in ("cost", "price"))


def hotel_keywords_for_budget(destination: str, budget_per_night: float | None) -> list[str]:
    """按每晚预算生成搜索关键词。

    这是"根据预算自动筛选"的第一层：**先让搜索引擎给出预算档位的候选**，
    再用解析出的价格做第二层过滤。只做第二层时，低价关键词往往搜不到高价酒店，
    反之却会把大量超预算结果塞进上下文。
    """
    city = normalize_city(destination)
    if budget_per_night is None:
        return [f"{city} 酒店", f"{city} 民宿"]
    if budget_per_night > 500:
        return [f"{city} 高端酒店", f"{city} 豪华酒店"]
    if budget_per_night > 300:
        return [f"{city} 酒店", f"{city} 品牌酒店"]
    return [f"{city} 经济型酒店", f"{city} 民宿"]


# --------------------------------------------------------------------------- 天气


def parse_weather_payload(payload: Any) -> tuple[list[WeatherDay], tuple[str, str] | None]:
    """解析多日预报。返回 `(按日期排序的预报, (最早, 最晚))`。"""
    if payload is None or isinstance(payload, str):
        return [], None
    items: list[Any] = []
    for candidate in _iter_lists(payload, ("forecasts", "casts", "forecast", "daily")):
        items.extend(candidate)
    if not items:
        for node in _iter_dicts(payload):
            if node.get("date") and (
                node.get("dayweather") or node.get("weather") or node.get("daytemp")
            ):
                items.append(node)

    days: list[WeatherDay] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        day = str(item.get("date") or item.get("day") or item.get("forecast_date") or "").strip()
        if not day or day in seen:
            continue
        seen.add(day)
        wind = " ".join(
            part
            for part in (
                str(item.get("daywind") or item.get("wind") or "").strip(),
                str(item.get("daypower") or "").strip(),
            )
            if part
        )
        days.append(
            WeatherDay(
                date=day,
                day_weather=str(
                    item.get("dayweather") or item.get("weather") or item.get("text") or ""
                ).strip(),
                night_weather=str(item.get("nightweather") or item.get("night_text") or "").strip(),
                day_temp=str(item.get("daytemp") or item.get("high") or item.get("temp_max") or "").strip(),
                night_temp=str(
                    item.get("nighttemp") or item.get("low") or item.get("temp_min") or ""
                ).strip(),
                wind=wind,
            )
        )
    days.sort(key=lambda d: d.date)
    if not days:
        return [], None
    return days, (days[0].date, days[-1].date)


# --------------------------------------------------------------------------- 黄历


def _first_text(node: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = node.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, (list, tuple)):
            return "、".join(str(v).strip() for v in value if str(v).strip())
        return str(value).strip()
    return ""


def parse_almanac_payload(payload: Any) -> Almanac | None:
    """解析黄历。不同服务商字段差异大（`公历/农历/干支/宜/忌` vs 英文键），
    因此选出"命中字段最多"的那个 dict，而不是假定固定层级。"""
    if payload is None or isinstance(payload, str):
        return None
    best: Mapping[str, Any] | None = None
    best_score = 0
    for node in _iter_dicts(payload):
        score = sum(
            1
            for key in ("公历", "农历", "干支", "宜", "忌", "yi", "ji", "lunar", "ganzhi")
            if key in node
        )
        if score > best_score:
            best, best_score = node, score
    if best is None or best_score == 0:
        return None

    solar = _first_text(best, ("公历", "solar_date", "solarDate", "solar", "date"))
    lunar = _first_text(best, ("农历", "lunar_date", "lunarDate", "lunar"))
    ganzhi = _first_text(best, ("干支", "ganzhi", "ganZhi", "gz_year"))
    yi = _first_text(best, ("宜", "yi", "good", "suitable"))
    ji = _first_text(best, ("忌", "ji", "bad", "avoid"))
    auspicious: bool | None = None
    if "出行" in yi or "出门" in yi:
        auspicious = True
    elif "出行" in ji or "出门" in ji:
        auspicious = False
    return Almanac(
        solar_date=solar,
        lunar_date=lunar,
        ganzhi=ganzhi,
        yi=yi,
        ji=ji,
        auspicious=auspicious,
        text="",
    )


# --------------------------------------------------------------------------- 门面


class LiveTravelData:
    """实时数据门面：业务语义（车次/路线/酒店/天气/黄历/航班）+ 解析 + 缓存。

    工具层只依赖这个门面；门面只依赖 `MCPToolHub` 的 `call/has`。
    测试里替换成假 hub 即可覆盖全部解析与降级分支，不需要起 MCP 服务。
    """

    def __init__(
        self,
        hub: Any,
        *,
        station_ttl_s: float = DEFAULT_STATION_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._hub = hub
        self._station_ttl_s = float(station_ttl_s)
        self._clock = clock
        self._station_cache: dict[str, tuple[float, list[StationCode]]] = {}

    # ------------------------------------------------------------------ 元信息
    @property
    def hub(self) -> Any:
        return self._hub

    def supports(self, server: str) -> bool:
        has = getattr(self._hub, "has", None)
        if callable(has):
            try:
                return bool(has(server))
            except Exception:  # noqa: BLE001
                return False
        names = getattr(self._hub, "server_names", ())
        return server in tuple(names)

    # ------------------------------------------------------------------ 底层调用
    async def _call_first(
        self,
        server: str,
        tool: str,
        variants: Sequence[Mapping[str, Any]],
        *,
        max_retries: int = 0,
        timeout_s: float | None = None,
    ) -> str:
        """按候选参数依次尝试。参数名在不同 MCP 服务实现里并不统一。"""
        errors: list[str] = []
        for arguments in variants:
            try:
                return await self._hub.call(
                    server,
                    tool,
                    dict(arguments),
                    max_retries=max_retries,
                    timeout_s=timeout_s,
                )
            except MCPError as exc:
                errors.append(f"{dict(arguments)} -> {exc}")
        detail = "；".join(errors[-2:]) if errors else "无可用参数组合"
        raise MCPError(
            f"{server}.{tool} 调用失败（{detail}）", server=server, tool=tool, retryable=False
        )

    @staticmethod
    def _clip(text: str) -> str:
        text = str(text or "")
        if len(text) <= MAX_TEXT_CHARS:
            return text
        return text[:MAX_TEXT_CHARS] + "\n…（结果过长，已截断）"

    # ------------------------------------------------------------------ 站点代码
    async def city_stations(self, city: str) -> list[StationCode]:
        """查询城市的所有车站代码，带 TTL 缓存。失败不缓存。"""
        key = normalize_city(city)
        if not key:
            return []
        cached = self._station_cache.get(key)
        now = self._clock()
        if cached is not None and now - cached[0] < self._station_ttl_s:
            return list(cached[1])

        variants: list[tuple[str, dict[str, Any]]] = [
            ("get-station-code-of-citys", {"citys": key}),
            ("get-stations-code-in-city", {"city": key}),
            ("get-station-code-of-citys", {"cities": key}),
        ]
        last_error: MCPError | None = None
        succeeded = False
        for tool, arguments in variants:
            try:
                text = await self._call_first(
                    TRAIN_SERVER, tool, [arguments], max_retries=0, timeout_s=10.0
                )
            except MCPError as exc:
                last_error = exc
                continue
            succeeded = True
            payload = _loads(text)
            stations = parse_station_payload(payload if payload is not None else text, key)
            if stations:
                self._station_cache[key] = (now, stations)
                return list(stations)
        if not succeeded and last_error is not None:
            raise last_error
        return []

    # ------------------------------------------------------------------ 火车票
    async def train_tickets(self, origin: str, destination: str, travel_date: str) -> TrainTickets:
        notes: list[str] = []
        from_code = to_code = None
        try:
            from_stations = await self.city_stations(origin)
            if from_stations:
                from_code = from_stations[0].code
        except MCPError as exc:
            notes.append(f"出发站代码查询失败：{exc}")
        try:
            to_stations = await self.city_stations(destination)
            if to_stations:
                to_code = to_stations[0].code
        except MCPError as exc:
            notes.append(f"到达站代码查询失败：{exc}")

        variants: list[dict[str, Any]] = []
        if from_code and to_code:
            variants.append(
                {"fromStation": from_code, "toStation": to_code, "date": travel_date}
            )
        # 站点代码拿不到时用城市名兜底：12306 的多数实现也接受城市名。
        variants.extend(
            [
                {"fromStation": origin, "toStation": destination, "date": travel_date},
                {"from": origin, "to": destination, "date": travel_date},
                {"departure": origin, "arrival": destination, "date": travel_date},
            ]
        )
        text = await self._call_first(
            TRAIN_SERVER, "get-tickets", variants, max_retries=1, timeout_s=18.0
        )
        return TrainTickets(
            origin=normalize_city(origin),
            destination=normalize_city(destination),
            date=travel_date,
            text=self._clip(text),
            from_code=from_code,
            to_code=to_code,
            stations_resolved=bool(from_code and to_code),
            notes=notes,
        )

    # ------------------------------------------------------------------ 地理编码
    async def geocode(self, address: str) -> str:
        text = await self._call_first(
            GAODE_SERVER,
            "maps_geo",
            [{"address": address}, {"city": address}],
            max_retries=0,
            timeout_s=12.0,
        )
        payload = _loads(text)
        coords = parse_geo_payload(payload if payload is not None else text)
        if coords is None:
            raise MCPError(
                f"高德未能解析「{address}」的经纬度", server=GAODE_SERVER, tool="maps_geo"
            )
        return coords

    # ------------------------------------------------------------------ 自驾路线
    async def driving_route(self, origin: str, destination: str) -> DrivingRoute:
        notes: list[str] = []
        try:
            origin_coords = await self.geocode(origin)
            destination_coords = await self.geocode(destination)
        except MCPError as exc:
            notes.append(f"地理编码失败：{exc}")
        else:
            try:
                text = await self._call_first(
                    GAODE_SERVER,
                    "maps_direction_driving",
                    [
                        {"origin": origin_coords, "destination": destination_coords},
                        {"origin": origin_coords, "destination": destination_coords, "strategy": 0},
                    ],
                    max_retries=0,
                    timeout_s=18.0,
                )
            except MCPError as exc:
                notes.append(f"驾车路线查询失败：{exc}")
            else:
                payload = _loads(text)
                facts = parse_route_payload(payload if payload is not None else text)
                if facts.distance_km is not None:
                    return DrivingRoute(
                        origin=normalize_city(origin),
                        destination=normalize_city(destination),
                        distance_km=facts.distance_km,
                        duration_minutes=facts.duration_minutes,
                        tolls_cny=facts.tolls_cny,
                        toll_distance_km=facts.toll_distance_km,
                        estimated=False,
                        notes=notes,
                    )
                notes.append("高德返回中未解析出距离字段，改用直线距离估算")

        estimate = estimate_distance(origin, destination)
        if estimate is None:
            raise MCPError(
                f"无法获取 {origin}→{destination} 的自驾距离；"
                + ("；".join(notes) if notes else "高德服务不可用"),
                server=GAODE_SERVER,
                tool="maps_direction_driving",
            )
        return DrivingRoute(
            origin=normalize_city(origin),
            destination=normalize_city(destination),
            distance_km=estimate.road_km,
            duration_minutes=max(1, int(round(estimate.road_km / 80 * 60))),
            tolls_cny=None,
            estimated=True,
            notes=notes + ["距离为直线距离 ×1.25 的估算值，过路费与实时路况需以高德为准"],
        )

    # ------------------------------------------------------------------ 酒店
    async def hotel_search(
        self,
        city: str,
        *,
        budget_per_night: float | None = None,
        keywords: Sequence[str] | None = None,
    ) -> HotelSearch:
        city_name = normalize_city(city)
        chosen = [str(k) for k in (keywords or []) if str(k).strip()]
        if not chosen:
            chosen = hotel_keywords_for_budget(city_name, budget_per_night)

        hits: list[HotelHit] = []
        texts: list[str] = []
        notes: list[str] = []
        structured = False
        for keyword in chosen[:2]:
            try:
                text = await self._call_first(
                    GAODE_SERVER,
                    "maps_text_search",
                    [{"keywords": keyword, "city": city_name}, {"keyword": keyword, "city": city_name}],
                    max_retries=0,
                    timeout_s=15.0,
                )
            except MCPError as exc:
                notes.append(f"「{keyword}」搜索失败：{exc}")
                continue
            payload = _loads(text)
            parsed = parse_poi_payload(payload if payload is not None else text)
            if parsed:
                structured = True
                hits.extend(parsed)
            texts.append(self._clip(text))

        # 去重保序
        seen: set[str] = set()
        unique: list[HotelHit] = []
        for hit in hits:
            if hit.name in seen:
                continue
            seen.add(hit.name)
            unique.append(hit)

        # 两个关键词都失败时不能假装"搜到了 0 家"，必须让调用方知道是服务故障。
        if not unique and not texts:
            raise MCPError(
                f"酒店搜索失败：{('；'.join(notes)) or '服务无返回'}",
                server=GAODE_SERVER,
                tool="maps_text_search",
                retryable=True,
            )

        over_budget = 0
        if budget_per_night is not None and unique:
            kept: list[HotelHit] = []
            for hit in unique:
                if hit.cost_cny is not None and hit.cost_cny > budget_per_night:
                    over_budget += 1
                    continue
                kept.append(hit)
            # 全部超预算时保留原列表并标注，而不是返回空：用户可能愿意微调预算。
            if kept:
                unique = kept

        return HotelSearch(
            city=city_name,
            keywords=chosen[:2],
            hits=unique,
            budget_per_night=budget_per_night,
            over_budget=over_budget,
            structured=structured,
            text="\n\n".join(texts),
            notes=notes,
        )

    # ------------------------------------------------------------------ 天气
    async def weather(
        self,
        city: str,
        *,
        start: date | None = None,
        days: int = 3,
    ) -> WeatherReport:
        city_name = normalize_city(city)
        text = await self._call_first(
            GAODE_SERVER,
            "maps_weather",
            [{"city": city_name}, {"city": city}, {"address": city_name}],
            max_retries=0,
            timeout_s=15.0,
        )
        payload = _loads(text)
        forecasts, available = parse_weather_payload(payload if payload is not None else text)
        structured = bool(forecasts)

        selected: list[WeatherDay] = []
        missing: list[str] = []
        if structured:
            if start is None:
                selected = forecasts[: max(1, days)]
            else:
                wanted = [start + timedelta(days=offset) for offset in range(max(1, days))]
                by_date = {day.date: day for day in forecasts}
                for want in wanted:
                    key = want.isoformat()
                    day = by_date.get(key)
                    if day is None:
                        missing.append(key)
                    else:
                        selected.append(day)
        return WeatherReport(
            city=city_name,
            days=selected,
            available_from=available[0] if available else "",
            available_to=available[1] if available else "",
            missing_dates=missing,
            structured=structured,
            text=self._clip(text),
        )

    # ------------------------------------------------------------------ 黄历
    async def almanac(self, day: date) -> Almanac:
        iso = day.isoformat()
        variants = [
            {"solarDatetime": f"{iso}T12:00:00+08:00"},
            {"date": iso},
            {"solarDate": iso},
        ]
        text = await self._call_first(
            BAZI_SERVER, "getChineseCalendar", variants, max_retries=0, timeout_s=12.0
        )
        payload = _loads(text)
        parsed = parse_almanac_payload(payload if payload is not None else text)
        if parsed is None:
            return Almanac(solar_date=iso, text=self._clip(text))
        if not parsed.solar_date:
            parsed.solar_date = iso
        parsed.text = self._clip(text)
        return parsed

    # ------------------------------------------------------------------ 航班
    async def flights(self, dep_city: str, arr_city: str, day: date) -> FlightSearch:
        dep_code = lookup_airport(dep_city)
        arr_code = lookup_airport(arr_city)
        if dep_code is None:
            raise AirportCodeError(dep_city)
        if arr_code is None:
            raise AirportCodeError(arr_city)
        iso = day.isoformat()
        text = await self._call_first(
            FLIGHT_SERVER,
            "searchFlightsByDepArr",
            [
                {"dep": dep_code, "arr": arr_code, "date": iso},
                {"dep": dep_code, "arr": arr_code, "date": iso.replace("-", "")},
            ],
            max_retries=0,
            timeout_s=15.0,
        )
        return FlightSearch(
            dep_city=normalize_city(dep_city),
            arr_city=normalize_city(arr_city),
            dep_code=dep_code,
            arr_code=arr_code,
            date=iso,
            text=self._clip(text),
        )


def known_city(city: str) -> bool:
    return normalize_city(city) in CITY_COORDS


__all__ = [
    "AirportCodeError",
    "Almanac",
    "DrivingRoute",
    "FlightSearch",
    "HotelHit",
    "HotelSearch",
    "LiveTravelData",
    "RouteFacts",
    "StationCode",
    "TrainTickets",
    "WeatherDay",
    "WeatherReport",
    "hotel_keywords_for_budget",
    "known_city",
    "parse_almanac_payload",
    "parse_geo_payload",
    "parse_poi_payload",
    "parse_pois_from_text",
    "parse_route_payload",
    "parse_station_payload",
    "parse_stations_from_text",
    "parse_weather_payload",
]
