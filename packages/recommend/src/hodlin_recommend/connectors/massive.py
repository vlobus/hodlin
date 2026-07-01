"""Massive price-bar connector (``PriceBarSource``) — OHLC aggregates (D13).

Turns Massive's aggregate JSON into ``PriceBar`` domain models for both stocks
and crypto. Dependencies are injected for testability.

NOTE: the exact Massive response shape is "verify on build" per D13. The
assumed schema is a single object with a ``bars`` array of
``{ts, open, high, low, close, volume}``; if the live API differs, ``_parse_bar``
and the ``bars`` key are the only things to adjust — the Protocol keeps that a
one-file change. Provider numbers are coerced Decimal-via-str so money never
passes through a float.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx

from hodlin_recommend.connectors.base import (
    DEFAULT_RETRY,
    RateLimiter,
    RetryPolicy,
    SourceUnavailable,
    request_json,
)
from hodlin_recommend.domain.models import PriceBar

_DATE_FMT = "%Y-%m-%d"


def _dec(value: Any) -> Decimal:
    """Coerce a provider number to exact Decimal without touching a float."""
    return Decimal(str(value))


def _parse_bar(symbol: str, interval: str, raw: Mapping[str, Any]) -> PriceBar:
    volume = raw.get("volume")
    return PriceBar(
        symbol=symbol,
        interval=interval,
        ts=datetime.fromisoformat(raw["ts"]).astimezone(UTC),
        open=_dec(raw["open"]),
        high=_dec(raw["high"]),
        low=_dec(raw["low"]),
        close=_dec(raw["close"]),
        volume=_dec(volume) if volume is not None else None,
        source="massive",
    )


class MassivePriceBarSource:
    source = "massive"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str,
        base_url: str,
        rate: RateLimiter,
        retry: RetryPolicy = DEFAULT_RETRY,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._rate = rate
        self._retry = retry

    async def get_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[PriceBar]:
        params = {
            "symbol": symbol,
            "interval": interval,
            "from": start.astimezone(UTC).strftime(_DATE_FMT),
            "to": end.astimezone(UTC).strftime(_DATE_FMT),
            "token": self._api_key,
        }
        payload = await request_json(
            self._client,
            source=self.source,
            url=f"{self._base_url}/aggregates",
            params=params,
            rate=self._rate,
            retry=self._retry,
        )
        bars = payload.get("bars") if isinstance(payload, Mapping) else None
        if not isinstance(bars, list):
            raise SourceUnavailable(self.source, "expected an object with a 'bars' array")
        return [_parse_bar(symbol, interval, raw) for raw in bars]

    async def health(self) -> bool:
        try:
            now = datetime.now(UTC)
            await self.get_candles("AAPL", "1d", now, now)
        except SourceUnavailable:
            return False
        return True
