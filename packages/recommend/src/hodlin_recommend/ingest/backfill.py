"""Cold-start backfill (D6): fill each asset's bar history, then detect.

On a fresh database the rolling z-score has no baseline, so the first live
tick could say nothing for ``window`` days. Backfill fetches ``backfill_days``
of candles per configured asset, upserts them (idempotent — re-running against
existing data inserts nothing), and runs one detection pass over the stored
history so any anomaly already in the data exists before the first tick.

An unavailable source skips that asset and records why, rather than failing
the whole cold start — the scheduler will retry on its normal cadence.
Backfill is an orchestration entry point, so unlike repositories it owns its
transaction and commits at the end.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from hodlin_recommend.connectors.base import PriceBarSource, SourceUnavailable
from hodlin_recommend.domain.anomaly import detect_series
from hodlin_recommend.domain.asset_config import AssetConfig
from hodlin_recommend.domain.models import Asset
from hodlin_recommend.store.repositories import (
    AnomalyRepository,
    AssetRepository,
    PriceBarRepository,
)


@dataclass(frozen=True)
class BackfillResult:
    """What backfill did for one asset — enough to log and to assert on."""

    symbol: str
    bars_inserted: int
    bars_stored: int
    anomalies_inserted: int
    skipped: str | None = None  # why the source was unavailable, if it was


async def backfill_assets(
    session: AsyncSession,
    source: PriceBarSource,
    assets: Sequence[AssetConfig],
    *,
    end: datetime | None = None,
) -> list[BackfillResult]:
    """Backfill + initial detection pass for every configured asset.

    ``end`` defaults to now; the seed-data demo passes a fixed ``end`` inside
    the CSV's date range, since seed bars are historical.
    """
    end = end if end is not None else datetime.now(UTC)
    results: list[BackfillResult] = []
    for config in assets:
        await AssetRepository(session).upsert(
            Asset(symbol=config.symbol, kind=config.kind, name=config.name)
        )
        try:
            bars = await source.get_candles(
                config.symbol,
                config.interval,
                end - timedelta(days=config.backfill_days),
                end,
            )
        except SourceUnavailable as exc:
            results.append(
                BackfillResult(
                    symbol=config.symbol,
                    bars_inserted=0,
                    bars_stored=0,
                    anomalies_inserted=0,
                    skipped=str(exc),
                )
            )
            continue
        bar_repo = PriceBarRepository(session)
        bars_inserted = await bar_repo.upsert_many(bars)
        # Detect over everything just fetched plus enough older stored history
        # to give the earliest fetched bar a full baseline. Sized off the fetch
        # itself, so it holds for any interval, not just daily bars.
        stored = await bar_repo.recent(
            config.symbol, config.interval, limit=len(bars) + config.window + 1
        )
        anomalies = detect_series(stored, window=config.window, threshold=config.threshold)
        anomalies_inserted = await AnomalyRepository(session).upsert_many(anomalies)
        results.append(
            BackfillResult(
                symbol=config.symbol,
                bars_inserted=bars_inserted,
                bars_stored=len(stored),
                anomalies_inserted=anomalies_inserted,
            )
        )
    await session.commit()
    return results
