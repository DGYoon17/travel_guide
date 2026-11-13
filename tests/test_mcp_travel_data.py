"""MCP 实时数据适配层测试：解析器 + 门面降级行为。

这里不打桩 pydantic、不启动 HTTP 服务，只验证两件最容易出错的事：
1. 不同服务商的返回形状都能被正确归一化（距离单位、预报日期、宜忌字段）；
2. 外部服务不可用时不抛栈、不编数值，而是给出可读的降级结果。
"""

from __future__ import annotations

from datetime import date

import pytest

from app.domain.travel.geo import estimate_distance, lookup_airport
from app.infrastructure.mcp.hub import BAZI_SERVER, FLIGHT_SERVER, GAODE_SERVER, TRAIN_SERVER, MCPError
from app.infrastructure.mcp.travel_data import (
    AirportCodeError,
    LiveTravelData,
    hotel_keywords_for_budget,
    parse_almanac_payload,
    parse_geo_payload,
    parse_poi_payload,
    parse_pois_from_text,
    parse_route_payload,
    parse_station_payload,
    parse_stations_from_text,
    parse_weather_payload,
)
from tests.mcp_fakes import (
    FakeMCPHub,
    almanac_handler,
    default_handlers,
    station_handler,
    weather_handler,
)


# --------------------------------------------------------------------------- 纯解析


def test_parse_station_payload_grouped_by_city() -> None:
    payload = {"上海": [{"station_name": "上海", "station_code": "SHH"}]}
    stations = parse_station_payload(payload, "上海")
    assert [s.code for s in stations] == ["SHH"]
    assert stations[0].name == "上海"


def test_parse_station_payload_nested_and_filtered() -> None:
    payload = {
        "data": [
            {"city": "上海", "stations": [{"name": "上海虹桥", "telecode": "AOH"}]},
            {"city": "杭州", "stations": [{"name": "杭州东", "telecode": "HGH"}]},
        ]
    }
    stations = parse_station_payload(payload, "上海市")
    assert [s.code for s in stations] == ["AOH"]


def test_parse_station_payload_empty_returns_empty() -> None:
    assert parse_station_payload({"error": "nope"}, "上海") == []


def test_parse_stations_from_text() -> None:
    stations = parse_stations_from_text("上海: SHH\n上海虹桥 AOH", "上海")
    assert stations[0].code == "SHH"
    assert {s.code for s in stations} == {"SHH", "AOH"}


def test_parse_geo_payload_nested() -> None:
    assert parse_geo_payload({"return": [{"location": "121.47,31.23"}]}) == "121.470000,31.230000"
    assert parse_geo_payload({"geocodes": [{"lng": 120.1, "lat": 30.2}]}) == "120.100000,30.200000"
    # 0,0 是无效坐标（高德在解析失败时会返回它），不能当成真实位置。
    assert parse_geo_payload({"location": "0,0"}) is None


def test_parse_route_payload_prefers_shortest_path() -> None:
    payload = {
        "route": {
            "paths": [
                {"distance": "190000", "duration": 9000, "tolls": "95"},
                {"distance": "176400", "duration": 8400, "tolls": "85"},
            ]
        }
    }
    facts = parse_route_payload(payload)
    assert facts.distance_km == 176.4
    assert facts.duration_minutes == 140
    assert facts.tolls_cny == 85


def test_parse_route_payload_accepts_km_and_minutes_fields() -> None:
    facts = parse_route_payload({"data": {"distance_km": 176.4, "duration_minutes": 140, "tolls": "85"}})
    assert facts.distance_km == 176.4
    assert facts.duration_minutes == 140
    assert facts.tolls_cny == 85


def test_parse_route_payload_handles_unit_text() -> None:
    facts = parse_route_payload({"distance": "176.4公里", "duration": "2小时20分", "tolls": "85元"})
    assert facts.distance_km == 176.4
    assert facts.duration_minutes == 140
    assert facts.tolls_cny == 85


def test_parse_pois_from_text() -> None:
    hits = parse_pois_from_text("1. 西湖民宿 人均¥280\n2. 豪华酒店 价格：￥1200\n地址：某路")
    assert [h.name for h in hits] == ["西湖民宿", "豪华酒店"]
    assert hits[0].cost_cny == 280
    assert hits[1].cost_cny == 1200


def test_parse_poi_payload_reads_biz_ext_cost() -> None:
    hits = parse_poi_payload({"pois": [{"name": "甲酒店", "biz_ext": {"cost": "320"}}]})
    assert hits and hits[0].cost_cny == 320


def test_hotel_keywords_for_budget() -> None:
    assert hotel_keywords_for_budget("杭州", 600)[0] == "杭州 高端酒店"
    assert hotel_keywords_for_budget("杭州", 400)[0] == "杭州 酒店"
    assert hotel_keywords_for_budget("杭州", 200)[0] == "杭州 经济型酒店"
    assert hotel_keywords_for_budget("杭州", None) == ["杭州 酒店", "杭州 民宿"]


def test_parse_weather_payload_sorted() -> None:
    days, span = parse_weather_payload(weather_handler({}))
    assert [d.date for d in days] == ["2026-11-20", "2026-11-21"]
    assert days[0].day_temp == "20"
    assert days[0].wind == "东北风 3级"
    assert span == ("2026-11-20", "2026-11-21")


def test_parse_almanac_payload_auspicious() -> None:
    almanac = parse_almanac_payload(almanac_handler({}))
    assert almanac is not None
    assert almanac.lunar_date == "十月初一"
    assert almanac.auspicious is True
    assert "会友" in almanac.yi


def test_parse_almanac_payload_avoid_travel() -> None:
    payload = {"data": {"公历": "2026-11-20", "农历": "十月初一", "宜": "祭祀", "忌": "出行,动土"}}
    almanac = parse_almanac_payload(payload)
    assert almanac is not None
    assert almanac.auspicious is False


def test_parse_almanac_payload_none_for_plain_text() -> None:
    assert parse_almanac_payload("今日宜出行") is None


def test_geo_reference_data() -> None:
    assert lookup_airport("上海市") == "SHA"
    assert lookup_airport("SZX") == "SZX"
    assert lookup_airport("不存在的城市") is None
    estimate = estimate_distance("上海", "杭州")
    assert estimate is not None
    assert estimate.road_km > estimate.straight_km


# --------------------------------------------------------------------------- 门面


def _hub(overrides: dict | None = None) -> FakeMCPHub:
    handlers = default_handlers()
    handlers.update(overrides or {})
    return FakeMCPHub(handlers)


async def test_facade_train_tickets_resolves_station_codes() -> None:
    hub = _hub()
    data = LiveTravelData(hub)
    tickets = await data.train_tickets("上海", "杭州", "2026-11-20")

    assert tickets.from_code == "SHH"
    assert tickets.to_code == "HZH"
    assert tickets.stations_resolved is True
    assert "G7301" in tickets.text
    ticket_call = hub.called(TRAIN_SERVER, "get-tickets")[-1]
    assert ticket_call == {"fromStation": "SHH", "toStation": "HZH", "date": "2026-11-20"}


async def test_facade_train_tickets_falls_back_to_city_names() -> None:
    def flaky_station(args: dict) -> str:
        city = str(args.get("citys") or args.get("city") or args.get("cities") or "")
        if city == "杭州":
            raise MCPError("站点服务不可用")
        return station_handler(args)

    hub = _hub(
        {
            (TRAIN_SERVER, "get-station-code-of-citys"): flaky_station,
            (TRAIN_SERVER, "get-stations-code-in-city"): flaky_station,
        }
    )
    tickets = await LiveTravelData(hub).train_tickets("上海", "杭州", "2026-11-20")

    assert tickets.from_code == "SHH"
    assert tickets.to_code is None
    assert tickets.stations_resolved is False
    assert tickets.notes  # 明确记录"到达站代码查询失败"
    ticket_call = hub.called(TRAIN_SERVER, "get-tickets")[-1]
    assert ticket_call["fromStation"] == "上海"
    assert ticket_call["toStation"] == "杭州"


async def test_facade_station_codes_are_cached() -> None:
    hub = _hub()
    data = LiveTravelData(hub)
    await data.city_stations("上海")
    await data.city_stations("上海")
    assert len(hub.called(TRAIN_SERVER, "get-station-code-of-citys")) == 1


async def test_facade_driving_route_parses_distance_time_tolls() -> None:
    route = await LiveTravelData(_hub()).driving_route("上海", "杭州")
    assert route.estimated is False
    assert route.distance_km == 176.4
    assert route.duration_minutes == 140
    assert route.tolls_cny == 85


async def test_facade_driving_route_degrades_to_estimate() -> None:
    hub = FakeMCPHub(default_handlers(), fail_servers={GAODE_SERVER})
    route = await LiveTravelData(hub).driving_route("上海", "杭州")

    assert route.estimated is True
    assert route.distance_km is not None and route.distance_km > 100
    # 估算模式下绝不编造过路费
    assert route.tolls_cny is None
    assert route.notes


async def test_facade_driving_route_unknown_city_raises() -> None:
    hub = FakeMCPHub(default_handlers(), fail_servers={GAODE_SERVER})
    with pytest.raises(MCPError):
        await LiveTravelData(hub).driving_route("上海", "不存在的地名")


async def test_facade_hotel_search_filters_over_budget() -> None:
    hub = _hub()
    search = await LiveTravelData(hub).hotel_search("杭州", budget_per_night=400)

    names = [hit.name for hit in search.hits]
    assert names == ["甲酒店（西湖店）", "丙民宿"]
    assert search.over_budget == 1
    assert search.budget_per_night == 400
    assert search.keywords[0] == "杭州 酒店"


async def test_facade_hotel_search_uses_budget_keywords() -> None:
    hub = _hub()
    await LiveTravelData(hub).hotel_search("杭州", budget_per_night=600)
    first_call = hub.called(GAODE_SERVER, "maps_text_search")[0]
    assert first_call["keywords"] == "杭州 高端酒店"
    assert first_call["city"] == "杭州"


async def test_facade_weather_matches_travel_dates() -> None:
    hub = _hub()
    report = await LiveTravelData(hub).weather("杭州", start=date(2026, 11, 20), days=3)

    assert [day.date for day in report.days] == ["2026-11-20", "2026-11-21"]
    assert report.missing_dates == ["2026-11-22"]
    assert report.available_to == "2026-11-21"
    assert report.structured is True


async def test_facade_almanac_uses_iso_datetime_first() -> None:
    hub = _hub()
    almanac = await LiveTravelData(hub).almanac(date(2026, 11, 20))

    assert almanac.lunar_date == "十月初一"
    assert almanac.auspicious is True
    first_call = hub.called(BAZI_SERVER, "getChineseCalendar")[0]
    assert first_call["solarDatetime"] == "2026-11-20T12:00:00+08:00"


async def test_facade_flights_converts_city_to_iata() -> None:
    hub = _hub()
    flight = await LiveTravelData(hub).flights("上海", "成都", date(2026, 11, 20))

    assert flight.dep_code == "SHA"
    assert flight.arr_code == "CTU"
    call = hub.called(FLIGHT_SERVER, "searchFlightsByDepArr")[0]
    assert call == {"dep": "SHA", "arr": "CTU", "date": "2026-11-20"}


async def test_facade_flights_unknown_city_raises_airport_error() -> None:
    with pytest.raises(AirportCodeError) as excinfo:
        await LiveTravelData(_hub()).flights("火星基地", "成都", date(2026, 11, 20))
    assert excinfo.value.city == "火星基地"
