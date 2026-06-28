"""Repositories — the only place SQL lives.

Each repository wraps an ``AsyncSession`` and exposes intent-named methods that
take/return pure domain models. Ingestion is idempotent: inserts use Postgres
``ON CONFLICT`` against the natural-key constraints declared in ``tables.py``,
so re-fetching the same data never duplicates a row. The caller owns the
transaction (commit/rollback) so multiple repository calls compose atomically.
"""

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from hodlin_recommend.domain.models import Asset, NewsItem, PriceBar
from hodlin_recommend.store import tables


class UnknownAsset(Exception):
    """Raised when a bar/news item references a symbol with no ``assets`` row.
    Assets must be upserted before data that references them."""

    def __init__(self, symbol: str) -> None:
        super().__init__(f"no asset row for symbol {symbol!r}; upsert it first")
        self.symbol = symbol


async def _resolve_asset_ids(session: AsyncSession, symbols: set[str]) -> dict[str, int]:
    """Map symbols to surrogate asset ids, raising if any symbol is unknown.
    Shared by the bar and news repositories — both reference assets by symbol."""
    stmt = select(tables.Asset.symbol, tables.Asset.id).where(tables.Asset.symbol.in_(symbols))
    found = {symbol: asset_id for symbol, asset_id in await session.execute(stmt)}
    missing = symbols - found.keys()
    if missing:
        raise UnknownAsset(sorted(missing)[0])
    return found


class AssetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, asset: Asset) -> int:
        """Insert the asset or update its mutable fields; return its id either
        way. ``returning`` gives us the id without a second round-trip."""
        stmt = (
            insert(tables.Asset)
            .values(symbol=asset.symbol, kind=asset.kind, name=asset.name)
            .on_conflict_do_update(
                index_elements=[tables.Asset.symbol],
                set_={"kind": asset.kind, "name": asset.name},
            )
            .returning(tables.Asset.id)
        )
        result = await self._session.execute(stmt)
        asset_id: int = result.scalar_one()
        return asset_id

    async def get_id(self, symbol: str) -> int | None:
        stmt = select(tables.Asset.id).where(tables.Asset.symbol == symbol)
        asset_id: int | None = await self._session.scalar(stmt)
        return asset_id


class PriceBarRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_many(self, bars: Sequence[PriceBar]) -> int:
        """Insert bars, skipping any whose natural key already exists. Returns
        the number of rows actually inserted (0 on a pure re-ingest)."""
        if not bars:
            return 0
        ids = await _resolve_asset_ids(self._session, {bar.symbol for bar in bars})
        rows = [
            {
                "asset_id": ids[bar.symbol],
                "interval": bar.interval,
                "ts": bar.ts,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "source": bar.source,
            }
            for bar in bars
        ]
        stmt = (
            insert(tables.PriceBar)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_price_bars_natural")
            .returning(tables.PriceBar.id)
        )
        inserted = (await self._session.scalars(stmt)).all()
        return len(inserted)

    async def recent(self, symbol: str, interval: str, limit: int) -> list[PriceBar]:
        """The newest ``limit`` bars for a symbol/interval, newest first — the
        read the rolling z-score consumes. Served by ``ix_price_bars_recent``."""
        stmt = (
            select(tables.PriceBar)
            .join(tables.Asset, tables.PriceBar.asset_id == tables.Asset.id)
            .where(tables.Asset.symbol == symbol, tables.PriceBar.interval == interval)
            .order_by(tables.PriceBar.ts.desc())
            .limit(limit)
        )
        rows = (await self._session.scalars(stmt)).all()
        return [
            PriceBar(
                symbol=symbol,
                interval=row.interval,
                ts=row.ts,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
                source=row.source,
            )
            for row in rows
        ]


class NewsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_many(self, items: Sequence[NewsItem]) -> int:
        """Insert news, skipping duplicates by ``(source, external_id)``.
        Returns the number of new rows inserted."""
        if not items:
            return 0
        ids = await _resolve_asset_ids(self._session, {item.symbol for item in items})
        rows = [
            {
                "asset_id": ids[item.symbol],
                "source": item.source,
                "external_id": item.external_id,
                "headline": item.headline,
                "url": item.url,
                "summary": item.summary,
                "published_at": item.published_at,
            }
            for item in items
        ]
        stmt = (
            insert(tables.NewsItem)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_news_items_natural")
            .returning(tables.NewsItem.id)
        )
        inserted = (await self._session.scalars(stmt)).all()
        return len(inserted)


class IngestRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def start(self, job: str) -> int:
        """Open an audit row for a job execution; returns its id to close later."""
        stmt = (
            insert(tables.IngestRun)
            .values(job=job, status="running")
            .returning(tables.IngestRun.id)
        )
        result = await self._session.execute(stmt)
        run_id: int = result.scalar_one()
        return run_id

    async def finish(
        self, run_id: int, *, status: str, items: int = 0, detail: str | None = None
    ) -> None:
        run = await self._session.get(tables.IngestRun, run_id)
        if run is None:
            return
        run.status = status
        run.items = items
        run.detail = detail
        run.finished_at = datetime.now(UTC)
