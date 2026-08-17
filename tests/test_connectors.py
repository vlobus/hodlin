"""Connector tests — all offline (respx mocks httpx; no network, no keys).

Covers the T4 acceptance: recorded JSON parses into domain models, and a forced
failure raises SourceUnavailable after backoff and is catchable without crashing.
"""

from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest
import respx
from hodlin_recommend.connectors.base import (
    NewsSource,
    PriceBarSource,
    RateLimiter,
    RetryPolicy,
    SourceUnavailable,
)
from hodlin_recommend.connectors.finnhub import FinnhubNewsSource, _parse_article
from hodlin_recommend.connectors.massive import MassivePriceBarSource, _parse_bar
from hodlin_recommend.connectors.seed_bars import SeedBarSource

_BASE = "https://api.test/v1"
_FAST = RetryPolicy(attempts=3, wait_initial=0.0, wait_max=0.0)


@pytest.fixture
def rate() -> RateLimiter:
    # Fresh per test: aiolimiter binds to the event loop it's created on, and
    # pytest-asyncio uses a new loop per test. (Production creates one at startup.)
    return RateLimiter(10_000)


# Recorded-shape fixtures ----------------------------------------------------

_FINNHUB_ARTICLE = {
    "category": "company",
    "datetime": 1717416000,  # 2024-06-03T12:00:00Z
    "headline": "Apple unveils something",
    "id": 7712345,
    "source": "Reuters",
    "summary": "A summary of the thing.",
    "url": "https://example.test/article",
}

_MASSIVE_BAR = {
    "ts": "2024-06-03T00:00:00Z",
    "open": 190.12,
    "high": 192.40,
    "low": 189.55,
    "close": 191.80,
    "volume": 55123000,
}


# Pure parsing ---------------------------------------------------------------


def test_finnhub_parse_article() -> None:
    item = _parse_article("AAPL", _FINNHUB_ARTICLE)
    assert item.symbol == "AAPL"
    assert item.source == "finnhub"
    assert item.external_id == "7712345"
    assert item.published_at == datetime(2024, 6, 3, 12, 0, tzinfo=UTC)
    assert item.url == "https://example.test/article"


def test_massive_parse_bar_keeps_decimal() -> None:
    bar = _parse_bar("AAPL", "1d", _MASSIVE_BAR)
    assert bar.close == Decimal("191.8")
    assert isinstance(bar.close, Decimal)
    assert bar.ts == datetime(2024, 6, 3, tzinfo=UTC)
    assert bar.source == "massive"


# HTTP happy paths (respx) ---------------------------------------------------


async def test_finnhub_get_news_parses_response(rate: RateLimiter) -> None:
    async with httpx.AsyncClient() as client, respx.mock:
        respx.get(f"{_BASE}/company-news").mock(
            return_value=httpx.Response(200, json=[_FINNHUB_ARTICLE, _FINNHUB_ARTICLE])
        )
        src = FinnhubNewsSource(client, api_key="k", base_url=_BASE, rate=rate, retry=_FAST)
        items = await src.get_news("AAPL", datetime(2024, 6, 1, tzinfo=UTC))
    assert len(items) == 2
    assert items[0].headline == "Apple unveils something"


async def test_massive_get_candles_parses_response(rate: RateLimiter) -> None:
    async with httpx.AsyncClient() as client, respx.mock:
        respx.get(f"{_BASE}/aggregates").mock(
            return_value=httpx.Response(200, json={"symbol": "AAPL", "bars": [_MASSIVE_BAR]})
        )
        src = MassivePriceBarSource(client, api_key="k", base_url=_BASE, rate=rate, retry=_FAST)
        now = datetime(2024, 6, 3, tzinfo=UTC)
        bars = await src.get_candles("AAPL", "1d", now, now)
    assert len(bars) == 1
    assert bars[0].open == Decimal("190.12")


# Forced failures -> SourceUnavailable ---------------------------------------


async def test_repeated_5xx_raises_source_unavailable_after_retries(rate: RateLimiter) -> None:
    async with httpx.AsyncClient() as client, respx.mock:
        route = respx.get(f"{_BASE}/company-news").mock(return_value=httpx.Response(503))
        src = FinnhubNewsSource(client, api_key="k", base_url=_BASE, rate=rate, retry=_FAST)
        with pytest.raises(SourceUnavailable) as excinfo:
            await src.get_news("AAPL", datetime(2024, 6, 1, tzinfo=UTC))
    assert route.call_count == _FAST.attempts  # retried, then gave up
    assert excinfo.value.source == "finnhub"


async def test_403_fails_fast_without_retry(rate: RateLimiter) -> None:
    async with httpx.AsyncClient() as client, respx.mock:
        route = respx.get(f"{_BASE}/company-news").mock(return_value=httpx.Response(403))
        src = FinnhubNewsSource(client, api_key="k", base_url=_BASE, rate=rate, retry=_FAST)
        with pytest.raises(SourceUnavailable):
            await src.get_news("AAPL", datetime(2024, 6, 1, tzinfo=UTC))
    assert route.call_count == 1  # 4xx is permanent — no retry storm


async def test_network_error_raises_source_unavailable(rate: RateLimiter) -> None:
    async with httpx.AsyncClient() as client, respx.mock:
        respx.get(f"{_BASE}/aggregates").mock(side_effect=httpx.ConnectError("boom"))
        src = MassivePriceBarSource(client, api_key="k", base_url=_BASE, rate=rate, retry=_FAST)
        now = datetime(2024, 6, 3, tzinfo=UTC)
        with pytest.raises(SourceUnavailable):
            await src.get_candles("AAPL", "1d", now, now)


async def test_retry_recovers_after_transient_error(rate: RateLimiter) -> None:
    async with httpx.AsyncClient() as client, respx.mock:
        respx.get(f"{_BASE}/company-news").mock(
            side_effect=[httpx.Response(503), httpx.Response(200, json=[_FINNHUB_ARTICLE])]
        )
        src = FinnhubNewsSource(client, api_key="k", base_url=_BASE, rate=rate, retry=_FAST)
        items = await src.get_news("AAPL", datetime(2024, 6, 1, tzinfo=UTC))
    assert len(items) == 1  # recovered on the retry


async def test_source_unavailable_is_catchable_by_caller(rate: RateLimiter) -> None:
    # A job would swallow this and degrade gracefully.
    async with httpx.AsyncClient() as client, respx.mock:
        respx.get(f"{_BASE}/company-news").mock(return_value=httpx.Response(500))
        src = FinnhubNewsSource(client, api_key="k", base_url=_BASE, rate=rate, retry=_FAST)
        degraded = False
        try:
            await src.get_news("AAPL", datetime(2024, 6, 1, tzinfo=UTC))
        except SourceUnavailable:
            degraded = True
        assert degraded


async def test_malformed_body_raises_source_unavailable(rate: RateLimiter) -> None:
    async with httpx.AsyncClient() as client, respx.mock:
        respx.get(f"{_BASE}/company-news").mock(return_value=httpx.Response(200, text="not json"))
        src = FinnhubNewsSource(client, api_key="k", base_url=_BASE, rate=rate, retry=_FAST)
        with pytest.raises(SourceUnavailable):
            await src.get_news("AAPL", datetime(2024, 6, 1, tzinfo=UTC))


async def test_missing_field_raises_source_unavailable(rate: RateLimiter) -> None:
    async with httpx.AsyncClient() as client, respx.mock:
        # article missing the required "id"/"datetime" keys
        respx.get(f"{_BASE}/company-news").mock(
            return_value=httpx.Response(200, json=[{"headline": "x"}])
        )
        src = FinnhubNewsSource(client, api_key="k", base_url=_BASE, rate=rate, retry=_FAST)
        with pytest.raises(SourceUnavailable):
            await src.get_news("AAPL", datetime(2024, 6, 1, tzinfo=UTC))


async def test_decimal_precision_survives_the_wire(rate: RateLimiter) -> None:
    # A JSON number with more digits than a float can hold must reach the model
    # exactly — proves parse_float=Decimal, not response.json()'s float parsing.
    body = (
        '{"bars":[{"ts":"2024-06-03T00:00:00Z","open":0.12345678901234567,'
        '"high":0.12345678901234567,"low":0.12345678901234567,'
        '"close":0.12345678901234567,"volume":1}]}'
    )
    async with httpx.AsyncClient() as client, respx.mock:
        respx.get(f"{_BASE}/aggregates").mock(return_value=httpx.Response(200, text=body))
        src = MassivePriceBarSource(client, api_key="k", base_url=_BASE, rate=rate, retry=_FAST)
        now = datetime(2024, 6, 3, tzinfo=UTC)
        bars = await src.get_candles("AAPL", "1d", now, now)
    assert bars[0].close == Decimal("0.12345678901234567")


# Seed fallback --------------------------------------------------------------


async def test_seed_source_parses_committed_csv() -> None:
    src = SeedBarSource()
    assert await src.health() is True
    bars = await src.get_candles(
        "BTC-USD", "1d", datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)
    )
    assert len(bars) >= 21  # enough for a z-score window + 1
    assert bars == sorted(bars, key=lambda b: b.ts)  # newest-last, ascending
    assert all(isinstance(b.close, Decimal) and b.source == "seed" for b in bars)


async def test_seed_source_filters_by_symbol_but_ignores_the_window() -> None:
    # A fixed demo fixture stands in for "recent history": it returns its whole
    # committed series regardless of the requested [start, end], so the
    # scheduled backfill (which asks for the last N days) finds it whatever the
    # wall-clock date. Symbol/interval still filter.
    src = SeedBarSource()
    narrow = await src.get_candles(
        "AAPL", "1d", datetime(2024, 6, 3, tzinfo=UTC), datetime(2024, 6, 5, tzinfo=UTC)
    )
    assert {b.symbol for b in narrow} == {"AAPL"}
    assert len(narrow) == 25  # the full AAPL series, not the 3-day window
    other = await src.get_candles(
        "AAPL", "1h", datetime(2024, 6, 3, tzinfo=UTC), datetime(2024, 6, 5, tzinfo=UTC)
    )
    assert other == []  # interval still filters


async def test_connectors_satisfy_protocols(rate: RateLimiter) -> None:
    async with httpx.AsyncClient() as client:
        news: NewsSource = FinnhubNewsSource(client, api_key="k", base_url=_BASE, rate=rate)
        price: PriceBarSource = MassivePriceBarSource(
            client, api_key="k", base_url=_BASE, rate=rate
        )
        seed: PriceBarSource = SeedBarSource()
    assert news.source == "finnhub"
    assert price.source == "massive"
    assert seed.source == "seed"
