"""城际距离与机场三字码参考数据。

实时数据工具在三种场景需要"不依赖外部服务的城市知识"：

1. **判断是否自动查航班**：12306/高德都可能慢或不可用，但">800km 才查航班"这条
   规则必须本地可判；否则航班查询的触发条件会变成"高德是否恰好可用"。
2. **机场三字码转换**：航班 MCP 只接受 IATA 代码，模型给的是中文城市名。
3. **高德不可用时的兜底提示**：给出直线距离，明确标注是估算，绝不编造过路费与时刻。

这些数据是**只读参考表**，不是事实来源：真实距离与过路费一律以高德返回为准，
本地只用于触发阈值与降级提示。生产环境应替换为城市主数据 / 机场主数据。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: 城市中心点（经度, 纬度）。用于直线距离估算，非路网距离。
CITY_COORDS: dict[str, tuple[float, float]] = {
    "北京": (116.4074, 39.9042),
    "上海": (121.4737, 31.2304),
    "广州": (113.2644, 23.1291),
    "深圳": (114.0579, 22.5431),
    "杭州": (120.1551, 30.2741),
    "南京": (118.7969, 32.0603),
    "苏州": (120.5853, 31.2989),
    "无锡": (120.3119, 31.4912),
    "常州": (119.9741, 31.8112),
    "徐州": (117.1848, 34.2618),
    "扬州": (119.4129, 32.3942),
    "南通": (120.8943, 31.9802),
    "宁波": (121.5440, 29.8683),
    "温州": (120.6994, 27.9944),
    "绍兴": (120.5820, 29.9971),
    "嘉兴": (120.7555, 30.7461),
    "湖州": (120.0868, 30.8943),
    "金华": (119.6474, 29.0792),
    "台州": (121.4208, 28.6564),
    "合肥": (117.2272, 31.8206),
    "福州": (119.2965, 26.0745),
    "厦门": (118.0894, 24.4798),
    "泉州": (118.5850, 24.9139),
    "南昌": (115.8582, 28.6829),
    "济南": (117.0009, 36.6758),
    "青岛": (120.3826, 36.0671),
    "烟台": (121.4479, 37.4638),
    "威海": (122.1204, 37.5131),
    "郑州": (113.6254, 34.7466),
    "洛阳": (112.4540, 34.6197),
    "开封": (114.3073, 34.7974),
    "武汉": (114.3055, 30.5928),
    "宜昌": (111.2865, 30.6919),
    "长沙": (112.9388, 28.2282),
    "张家界": (110.4790, 29.1171),
    "常德": (111.6985, 29.0316),
    "成都": (104.0665, 30.5723),
    "重庆": (106.5049, 29.5332),
    "西安": (108.9398, 34.3416),
    "昆明": (102.8329, 24.8801),
    "大理": (100.2676, 25.6065),
    "丽江": (100.2330, 26.8721),
    "西双版纳": (100.7979, 22.0017),
    "贵阳": (106.6302, 26.6477),
    "南宁": (108.3665, 22.8170),
    "桂林": (110.2902, 25.2736),
    "北海": (109.1194, 21.4733),
    "海口": (110.1983, 20.0440),
    "三亚": (109.5119, 18.2528),
    "天津": (117.1901, 39.1252),
    "石家庄": (114.5149, 38.0428),
    "太原": (112.5489, 37.8706),
    "呼和浩特": (111.7519, 40.8414),
    "沈阳": (123.4315, 41.8057),
    "大连": (121.6147, 38.9140),
    "长春": (125.3235, 43.8171),
    "哈尔滨": (126.5349, 45.8038),
    "兰州": (103.8343, 36.0611),
    "西宁": (101.7782, 36.6171),
    "银川": (106.2309, 38.4872),
    "乌鲁木齐": (87.6168, 43.8256),
    "喀什": (75.9898, 39.4704),
    "库尔勒": (86.1746, 41.7259),
    "拉萨": (91.1409, 29.6456),
    "敦煌": (94.6619, 40.1421),
    "秦皇岛": (119.6005, 39.9354),
    "唐山": (118.1802, 39.6309),
    "东莞": (113.7518, 23.0207),
    "佛山": (113.1220, 23.0288),
    "珠海": (113.5767, 22.2707),
    "中山": (113.3926, 22.5170),
    "惠州": (114.4161, 23.1115),
    "汕头": (116.6822, 23.3535),
    "湛江": (110.3594, 21.2707),
    "黄山": (118.3376, 29.7147),
    "义乌": (120.0751, 29.3069),
    "舟山": (122.2072, 29.9853),
    "香港": (114.1694, 22.3193),
    "澳门": (113.5439, 22.1987),
    "台北": (121.5654, 25.0330),
    "东京": (139.6917, 35.6895),
    "大阪": (135.5023, 34.6937),
    "首尔": (126.9780, 37.5665),
    "新加坡": (103.8198, 1.3521),
    "曼谷": (100.5018, 13.7563),
    "伦敦": (-0.1276, 51.5072),
    "巴黎": (2.3522, 48.8566),
    "纽约": (-74.0060, 40.7128),
    "旧金山": (-122.4194, 37.7749),
    "洛杉矶": (-118.2437, 34.0522),
}

#: 城市 -> 主要机场 IATA 三字码。一个城市只取主力机场；多机场城市由供应方自行分流。
CITY_AIRPORTS: dict[str, str] = {
    "北京": "PEK",
    "上海": "SHA",
    "广州": "CAN",
    "深圳": "SZX",
    "成都": "CTU",
    "杭州": "HGH",
    "重庆": "CKG",
    "西安": "XIY",
    "武汉": "WUH",
    "南京": "NKG",
    "青岛": "TAO",
    "大连": "DLC",
    "天津": "TSN",
    "厦门": "XMN",
    "福州": "FOC",
    "昆明": "KMG",
    "长沙": "CSX",
    "济南": "TNA",
    "郑州": "CGO",
    "沈阳": "SHE",
    "哈尔滨": "HRB",
    "长春": "CGQ",
    "南昌": "KHN",
    "拉萨": "LXA",
    "乌鲁木齐": "URC",
    "海口": "HAK",
    "三亚": "SYX",
    "贵阳": "KWE",
    "银川": "INC",
    "兰州": "LHW",
    "太原": "TYN",
    "石家庄": "SJW",
    # 苏州没有民用机场，最近的是无锡硕放（WUX），按邻近机场换算而不是编一个不存在的代码。
    "苏州": "WUX",
    "无锡": "WUX",
    "宁波": "NGB",
    "温州": "WNZ",
    "南通": "NTG",
    "常州": "CZX",
    "徐州": "XUZ",
    "扬州": "YTY",
    "大理": "DLU",
    "丽江": "LJG",
    "威海": "WEH",
    "烟台": "YNT",
    "唐山": "TVS",
    "南宁": "NNG",
    "桂林": "KWL",
    "北海": "BHY",
    "黄山": "TXN",
    "张家界": "DYG",
    "常德": "CGD",
    "宜昌": "YIH",
    "敦煌": "DNH",
    "秦皇岛": "BPE",
    "呼和浩特": "HET",
    "西宁": "XNN",
    "洛阳": "LYA",
    "汕头": "SWA",
    "湛江": "ZHA",
    "珠海": "ZUH",
    "惠州": "HUZ",
    "佛山": "FUO",
    "泉州": "JJN",
    "义乌": "YIW",
    "舟山": "HSN",
    "西双版纳": "JHG",
    "香港": "HKG",
    "澳门": "MFM",
    "台北": "TPE",
    "东京": "NRT",
    "大阪": "KIX",
    "首尔": "ICN",
    "新加坡": "SIN",
    "曼谷": "BKK",
    "伦敦": "LHR",
    "巴黎": "CDG",
    "纽约": "JFK",
    "旧金山": "SFO",
    "洛杉矶": "LAX",
}

#: 直飞/高铁里程换算系数：路网距离通常约为直线距离的 1.2~1.3 倍。
ROAD_FACTOR = 1.25

#: 城市名前缀里常见但与城市名无关的后缀，归一化时剥离。
_CITY_SUFFIXES: tuple[str, ...] = ("特别行政区", "自治区", "地区", "自治州", "市", "区", "县")

EARTH_RADIUS_KM = 6371.0088


def normalize_city(name: str) -> str:
    """把「上海市」「北京 市」归一化成「上海」「北京」，便于查表。

    不做同义词扩展（"帝都"→北京）：那是主数据/模型的职责，代码里硬编码同义词
    会让"假设"看起来像"事实"。
    """
    text = str(name or "").strip().replace(" ", "").replace("　", "")
    for suffix in _CITY_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
            break
    return text


def _coord(city: str) -> tuple[float, float] | None:
    text = normalize_city(city)
    if text in CITY_COORDS:
        return CITY_COORDS[text]
    # 「上海虹桥」「北京首都机场」这类输入：取最长可匹配的城市前缀。
    for known in sorted(CITY_COORDS, key=len, reverse=True):
        if text.startswith(known):
            return CITY_COORDS[known]
    return None


def haversine_km(origin: tuple[float, float], destination: tuple[float, float]) -> float:
    lon1, lat1 = origin
    lon2, lat2 = destination
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


@dataclass(frozen=True)
class DistanceEstimate:
    """直线距离估算结果。`approximate=True` 是硬约束：调用方必须如实标注。"""

    origin: str
    destination: str
    straight_km: float
    road_km: float
    approximate: bool = True


def estimate_distance(origin: str, destination: str) -> DistanceEstimate | None:
    """按城市中心点估算距离；任一城市不在表内则返回 None（不猜）。"""
    a = _coord(origin)
    b = _coord(destination)
    if a is None or b is None:
        return None
    straight = haversine_km(a, b)
    return DistanceEstimate(
        origin=normalize_city(origin),
        destination=normalize_city(destination),
        straight_km=round(straight, 1),
        road_km=round(straight * ROAD_FACTOR, 1),
    )


def lookup_airport(city: str) -> str | None:
    """城市名 -> IATA 三字码。已经是三字码的输入原样返回（大写）。"""
    text = normalize_city(city)
    if not text:
        return None
    if len(text) == 3 and text.isascii() and text.isalpha():
        return text.upper()
    if text in CITY_AIRPORTS:
        return CITY_AIRPORTS[text]
    for known in sorted(CITY_AIRPORTS, key=len, reverse=True):
        if text.startswith(known):
            return CITY_AIRPORTS[known]
    return None


def looks_like_airport_code(text: str) -> bool:
    value = str(text or "").strip()
    return len(value) == 3 and value.isascii() and value.isalpha() and value.upper() == value


__all__ = [
    "CITY_AIRPORTS",
    "CITY_COORDS",
    "EARTH_RADIUS_KM",
    "ROAD_FACTOR",
    "DistanceEstimate",
    "estimate_distance",
    "haversine_km",
    "looks_like_airport_code",
    "lookup_airport",
    "normalize_city",
]
