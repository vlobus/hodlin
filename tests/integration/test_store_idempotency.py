"""The T3 acceptance test: ingesting the same natural key twice yields one row.

Proves the ON CONFLICT upserts honour the natural-key constraints, so a
re-run of ingestion never duplicates data.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from hodlin_recommend.domain.models import Asset, NewsItem, PriceBar
from hodlin_recommend.store.repositories import (
    AssetRepository,
    NewsRepository,
    PriceBarRepository,
    UnknownAsset,
)
from sqlalchemy.ext.asyncio import AsyncSession


def _bar() -> PriceBar:
    return PriceBar(
        symbol="BTC-USD",
        interval="1d",
        ts=datetime(2026, 6, 28, tzinfo=UTC),
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("95"),
        close=Decimal("105"),
        volume=Decimal("12.5"),
        source="seed",
    )


async def test_price_bar_upsert_is_idempotent(session: AsyncSession) -> None:
    await AssetRepository(session).upsert(Asset(symbol="BTC-USD", kind="crypto"))
    repo = PriceBarRepository(session)

    first = await repo.upsert_many([_bar()])
    second = await repo.upsert_many([_bar()])
    await session.commit()

    assert first == 1  # inserted
    assert second == 0  # same natural key -> skipped
    rows = await repo.recent("BTC-USD", "1d", limit=10)
    assert len(rows) == 1
    assert rows[0].close == Decimal("105")


async def test_asset_upsert_returns_same_id(session: AsyncSession) -> None:
    repo = AssetRepository(session)
    first_id = await repo.upsert(Asset(symbol="AAPL", kind="stock", name="Apple"))
    second_id = await repo.upsert(Asset(symbol="AAPL", kind="stock", name="Apple Inc."))
    await session.commit()

    assert first_id == second_id
    assert await repo.get_id("AAPL") == first_id


async def test_news_upsert_dedupes_on_source_external_id(session: AsyncSession) -> None:
    await AssetRepository(session).upsert(Asset(symbol="AAPL", kind="stock"))
    repo = NewsRepository(session)
    item = NewsItem(
        symbol="AAPL",
        source="finnhub",
        external_id="news-1",
        headline="Apple announces something",
        published_at=datetime(2026, 6, 28, 9, 30, tzinfo=UTC),
    )

    assert await repo.upsert_many([item]) == 1
    assert await repo.upsert_many([item]) == 0
    await session.commit()


async def test_bar_for_unknown_asset_raises(session: AsyncSession) -> None:
    repo = PriceBarRepository(session)
    with pytest.raises(UnknownAsset):
        await repo.upsert_many([_bar()])
