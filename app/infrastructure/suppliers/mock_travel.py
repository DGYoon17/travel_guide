"""确定性的差旅供应方 mock。

设计目标不是"像真的一样随机"，而是**可复现**：
同样的航线、日期、舱位，永远得到同一组报价，评测与回归才有意义。
价格变动、售罄这类异常由测试显式触发，不靠随机数碰运气。
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Protocol

from app.domain.travel.models import (
    BookingRecord,
    BookingStatus,
    CabinClass,
    OfferKind,
    TravelOffer,
)


class SupplierError(RuntimeError):
    """供应方受控错误。编排层会把它转成用户可读的失败原因。"""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class TravelSupplier(Protocol):
    async def search_flights(
        self,
        *,
        origin_city: str,
        destination_city: str,
        depart_date: date,
        return_date: date | None = None,
        cabin_class: CabinClass = CabinClass.ECONOMY,
        max_price_cny: Decimal | None = None,
    ) -> list[TravelOffer]: ...

    async def search_hotels(
        self,
        *,
        city: str,
        check_in: date,
        check_out: date,
        star: int | None = None,
        max_price_cny: Decimal | None = None,
    ) -> list[TravelOffer]: ...

    async def quote(self, offer_id: str) -> TravelOffer: ...

    async def book(
        self,
        offer_id: str,
        *,
        trip_id: str,
        traveler_id: str,
        idempotency_key: str,
        tenant_id: str = "",
    ) -> BookingRecord: ...

    async def cancel(self, booking_id: str, *, reason: str = "") -> BookingRecord: ...

    async def get_booking(self, booking_id: str) -> BookingRecord | None: ...


_CABIN_MULTIPLIER: dict[CabinClass, Decimal] = {
    CabinClass.ECONOMY: Decimal("1.0"),
    CabinClass.PREMIUM_ECONOMY: Decimal("1.6"),
    CabinClass.BUSINESS: Decimal("3.2"),
    CabinClass.FIRST: Decimal("4.8"),
}

_AIRLINES = (
    ("中国国航", "CA"),
    ("东方航空", "MU"),
    ("南方航空", "CZ"),
)

_HOTELS = (
    ("亚朵酒店", 4, "商务大床房"),
    ("全季酒店", 3, "高级大床房"),
    ("希尔顿欢朋", 4, "舒适双床房"),
)

_DEPART_TIMES = ("08:05", "13:30", "19:10")


def _seed(*parts: object) -> int:
    raw = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10], 16)


def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


class MockTravelSupplier:
    """进程内 mock 供应方。报价、订单与幂等键都保存在实例里。"""

    def __init__(self, *, offer_ttl_minutes: int = 30) -> None:
        self._ttl = timedelta(minutes=offer_ttl_minutes)
        self._offers: dict[str, TravelOffer] = {}
        self._bookings: dict[str, BookingRecord] = {}
        self._idempotency: dict[str, str] = {}
        self._price_override: dict[str, Decimal] = {}
        self._sold_out: set[str] = set()

    # ------------------------------------------------------------------ 查询
    async def search_flights(
        self,
        *,
        origin_city: str,
        destination_city: str,
        depart_date: date,
        return_date: date | None = None,
        cabin_class: CabinClass = CabinClass.ECONOMY,
        max_price_cny: Decimal | None = None,
    ) -> list[TravelOffer]:
        offers = self._flight_offers(
            origin_city, destination_city, depart_date, cabin_class, direction="outbound"
        )
        if return_date is not None:
            offers.extend(
                self._flight_offers(
                    destination_city, origin_city, return_date, cabin_class, direction="return"
                )
            )
        if max_price_cny is not None:
            offers = [o for o in offers if o.price_cny <= max_price_cny]
        for offer in offers:
            self._offers[offer.offer_id] = offer
        return offers

    async def search_hotels(
        self,
        *,
        city: str,
        check_in: date,
        check_out: date,
        star: int | None = None,
        max_price_cny: Decimal | None = None,
    ) -> list[TravelOffer]:
        nights = max(1, (check_out - check_in).days)
        offers: list[TravelOffer] = []
        for index, (name, default_star, room) in enumerate(_HOTELS):
            hotel_star = star or default_star
            seed = _seed(city, check_in, name)
            nightly = Decimal("260") + Decimal(seed % 420) + Decimal(hotel_star) * Decimal("120")
            offer = TravelOffer(
                offer_id=f"htl_{seed:010x}",
                kind=OfferKind.HOTEL,
                supplier=name,
                title=f"{city}·{name} {hotel_star}星 {room}",
                price_cny=_money(nightly * nights),
                expires_at=datetime.now(timezone.utc) + self._ttl,
                refundable=hotel_star >= 4,
                cancellation_policy="入住前 24 小时可免费取消" if hotel_star >= 4 else "预订后不可取消",
                details={
                    "city": city,
                    "star": hotel_star,
                    "room": room,
                    "check_in": check_in.isoformat(),
                    "check_out": check_out.isoformat(),
                    "nights": nights,
                    "nightly_cny": str(_money(nightly)),
                },
            )
            offers.append(offer)
            self._offers[offer.offer_id] = offer
        if max_price_cny is not None:
            offers = [o for o in offers if o.price_cny <= max_price_cny]
        return offers

    def _flight_offers(
        self,
        origin: str,
        destination: str,
        depart_date: date,
        cabin: CabinClass,
        *,
        direction: str,
    ) -> list[TravelOffer]:
        offers: list[TravelOffer] = []
        for index, (airline, code) in enumerate(_AIRLINES):
            seed = _seed(origin, destination, depart_date, code, direction)
            duration = 90 + seed % 240
            base = Decimal("680") + Decimal(seed % 900)
            price = _money(base * _CABIN_MULTIPLIER[cabin])
            dep_hour, dep_minute = (int(x) for x in _DEPART_TIMES[index].split(":"))
            arrive_total = dep_hour * 60 + dep_minute + duration
            offers.append(
                TravelOffer(
                    offer_id=f"flt_{seed:010x}",
                    kind=OfferKind.FLIGHT,
                    supplier=airline,
                    title=(
                        f"{origin}→{destination} {code}{1000 + seed % 9000} "
                        f"{_DEPART_TIMES[index]} 起飞，约 {duration // 60}小时{duration % 60}分"
                    ),
                    price_cny=price,
                    expires_at=datetime.now(timezone.utc) + self._ttl,
                    refundable=cabin in {CabinClass.BUSINESS, CabinClass.FIRST} or price > Decimal("1500"),
                    cancellation_policy="起飞前 4 小时可退，收取 20% 手续费"
                    if price > Decimal("1500")
                    else "特价票，仅退机建燃油",
                    details={
                        "direction": direction,
                        "flight_no": f"{code}{1000 + seed % 9000}",
                        "airline": airline,
                        "depart_time": _DEPART_TIMES[index],
                        "arrive_time": f"{(arrive_total // 60) % 24:02d}:{arrive_total % 60:02d}",
                        "duration_minutes": duration,
                        "cabin": cabin.value,
                        "date": depart_date.isoformat(),
                    },
                )
            )
            self._offers[offers[-1].offer_id] = offers[-1]
        return sorted(offers, key=lambda o: o.price_cny)

    # ------------------------------------------------------------------ 报价
    async def quote(self, offer_id: str) -> TravelOffer:
        offer = self._offers.get(offer_id)
        if offer is None:
            raise SupplierError("OFFER_NOT_FOUND", f"报价 {offer_id} 不存在或已过期，请重新搜索。")
        if offer.expires_at <= datetime.now(timezone.utc):
            raise SupplierError("OFFER_EXPIRED", f"报价 {offer_id} 已过期，请重新搜索。")
        override = self._price_override.get(offer_id)
        if override is not None:
            return offer.model_copy(update={"price_cny": _money(override)})
        return offer

    # ------------------------------------------------------------------ 预订
    async def book(
        self,
        offer_id: str,
        *,
        trip_id: str,
        traveler_id: str,
        idempotency_key: str,
        tenant_id: str = "",
    ) -> BookingRecord:
        if idempotency_key and idempotency_key in self._idempotency:
            existing = self._bookings.get(self._idempotency[idempotency_key])
            if existing is not None:
                return existing
        if offer_id in self._sold_out:
            raise SupplierError("SOLD_OUT", "该报价已售罄，请选择其它班次或房型。")
        offer = await self.quote(offer_id)
        booking_id = f"bk_{uuid.uuid4().hex[:12]}"
        order_no = (
            f"PNR{uuid.uuid4().hex[:8].upper()}"
            if offer.kind is OfferKind.FLIGHT
            else f"HTL{uuid.uuid4().hex[:8].upper()}"
        )
        record = BookingRecord(
            booking_id=booking_id,
            tenant_id=tenant_id,
            trip_id=trip_id,
            offer_id=offer_id,
            kind=offer.kind,
            status=BookingStatus.CONFIRMED,
            supplier=offer.supplier,
            title=offer.title,
            price_cny=offer.price_cny,
            traveler_id=traveler_id,
            order_no=order_no,
            idempotency_key=idempotency_key,
            created_at=datetime.now(timezone.utc),
        )
        self._bookings[booking_id] = record
        if idempotency_key:
            self._idempotency[idempotency_key] = booking_id
        return record

    async def cancel(self, booking_id: str, *, reason: str = "") -> BookingRecord:
        record = self._bookings.get(booking_id)
        if record is None:
            raise SupplierError("BOOKING_NOT_FOUND", f"订单 {booking_id} 不存在。")
        if record.status is BookingStatus.CANCELLED:
            return record
        fee = Decimal("0") if record.status is not BookingStatus.CONFIRMED or record.price_cny < Decimal("1000") else _money(record.price_cny * Decimal("0.2"))
        updated = record.model_copy(
            update={
                "status": BookingStatus.CANCELLED,
                "note": f"取消原因：{reason or '未说明'}；退票费 {fee} 元",
            }
        )
        self._bookings[booking_id] = updated
        return updated

    async def get_booking(self, booking_id: str) -> BookingRecord | None:
        return self._bookings.get(booking_id)

    # ------------------------------------------------------------------ 测试钩子
    def simulate_price_change(self, offer_id: str, *, price_cny: Decimal) -> None:
        """把某个报价改成新价格，用于验证确认门的二次校验。"""
        self._price_override[offer_id] = price_cny

    def simulate_sold_out(self, offer_id: str) -> None:
        self._sold_out.add(offer_id)

    def reset(self) -> None:
        self._offers.clear()
        self._bookings.clear()
        self._idempotency.clear()
        self._price_override.clear()
        self._sold_out.clear()
