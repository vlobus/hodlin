"""The T5 acceptance test: cold-start backfill against a real Postgres.

Backfill from the seed source must leave >= window+1 bars per asset, the
initial detection pass must surface the demo anomaly, and a re-run must be a
no-op (idempotent bars and anomalies). Also proves one dead source degrades
that asset instead of failing the whole cold start.
"""

from datetime import UTC, datetime

from hodlin_recommend.connectors.base import PriceBarSource, SourceUnavailable
from hodlin_recommend.connectors.seed_bars import SeedBarSource
from hodlin_recommend.domain.asset_config import AssetConfig
from hodlin_recommend.domain.models import PriceBar
from hodlin_recommend.ingest.backfill import backfill_assets
from hodlin_recommend.store.repositories import AnomalyRepository
from sqlalchemy.ext.asyncio import AsyncSession

# Inside the committed CSV's date range — seed bars are historical.
_END = datetime(2024, 7, 1, tzinfo=UTC)

_ASSETS = [
    AssetConfig(symbol="AAPL", kind="stock", name="Apple Inc."),
    AssetConfig(symbol="BTC-USD", kind="crypto", name="Bitcoin / USD"),
]


class _DeadSource:
    """A PriceBarSource whose provider is down — every fetch fails."""

    source = "dead"

    async def get_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[PriceBar]:
        raise SourceUnavailable(self.source, "connection refused")

    async def health(self) -> bool:
        return False


async def test_backfill_fills_window_and_detects_demo_anomaly(session: AsyncSession) -> None:
    results = await backfill_assets(session, SeedBarSource(), _ASSETS, end=_END)

    by_symbol = {result.symbol: result for result in results}
    for config in _ASSETS:
        result = by_symbol[config.symbol]
        assert result.skipped is None
        assert result.bars_stored >= config.window + 1
        assert result.bars_inserted == result.bars_stored  # fresh DB: all new

    # The seed data's late-June BTC drop trips at the demo config; AAPL stays
    # quiet at the same threshold — the z-score calibrates per asset.
    assert by_symbol["BTC-USD"].anomalies_inserted >= 1
    assert by_symbol["AAPL"].anomalies_inserted == 0

    stored = await AnomalyRepository(session).for_symbol("BTC-USD", "1d")
    assert len(stored) == by_symbol["BTC-USD"].anomalies_inserted
    assert stored[0].direction == "down"


async def test_backfill_rerun_is_idempotent(session: AsyncSession) -> None:
    await backfill_assets(session, SeedBarSource(), _ASSETS, end=_END)
    rerun = await backfill_assets(session, SeedBarSource(), _ASSETS, end=_END)

    for result in rerun:
        assert result.bars_inserted == 0
        assert result.anomalies_inserted == 0
        assert result.bars_stored >= 1  # history still there, nothing duplicated


async def test_dead_source_skips_asset_without_failing_cold_start(
    session: AsyncSession,
) -> None:
    source: PriceBarSource = _DeadSource()
    results = await backfill_assets(session, source, _ASSETS)

    assert [result.symbol for result in results] == ["AAPL", "BTC-USD"]
    for result in results:
        assert result.skipped is not None
        assert result.bars_stored == 0
