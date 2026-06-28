"""Unit checks for the store that need no database — schema registration and
the pure domain models. The DB behaviour (idempotency) is covered by the
testcontainers integration tests.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from hodlin_recommend.domain.models import Asset, NewsItem, PriceBar
from hodlin_recommend.store.db import Base
from pydantic import ValidationError

_EXPECTED_TABLES = {
    "assets",
    "price_bars",
    "news_items",
    "sentiments",
    "anomalies",
    "explanations",
    "source_health",
    "ingest_runs",
}


def test_metadata_registers_all_tables() -> None:
    assert set(Base.metadata.tables) == _EXPECTED_TABLES


def test_price_bars_natural_key_constraint_exists() -> None:
    constraints = {c.name for c in Base.metadata.tables["price_bars"].constraints}
    assert "uq_price_bars_natural" in constraints


def test_price_bar_keeps_money_as_decimal() -> None:
    bar = PriceBar(
        symbol="BTC-USD",
        interval="1d",
        ts=datetime(2026, 6, 28, tzinfo=UTC),
        open=Decimal("1"),
        high=Decimal("2"),
        low=Decimal("1"),
        close=Decimal("1.5"),
        source="seed",
    )
    assert bar.close == Decimal("1.5")
    assert bar.volume is None


def test_domain_models_are_frozen() -> None:
    asset = Asset(symbol="AAPL", kind="stock")
    with pytest.raises(ValidationError):
        asset.symbol = "MSFT"


def test_price_bar_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError):
        PriceBar(
            symbol="BTC-USD",
            interval="1d",
            ts=datetime(2026, 6, 28),  # intentionally naive
            open=Decimal("1"),
            high=Decimal("1"),
            low=Decimal("1"),
            close=Decimal("1"),
            source="seed",
        )


def test_price_bar_rejects_float_price() -> None:
    with pytest.raises(ValidationError):
        PriceBar(
            symbol="BTC-USD",
            interval="1d",
            ts=datetime(2026, 6, 28, tzinfo=UTC),
            open=1.5,  # type: ignore[arg-type]
            high=Decimal("1"),
            low=Decimal("1"),
            close=Decimal("1"),
            source="seed",
        )


def test_news_item_optional_fields_default_none() -> None:
    item = NewsItem(
        symbol="AAPL",
        source="finnhub",
        external_id="abc123",
        headline="something happened",
        published_at=datetime(2026, 6, 28, tzinfo=UTC),
    )
    assert item.url is None and item.summary is None
