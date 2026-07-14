"""Repositories — the only place SQL lives.

Each repository wraps an ``AsyncSession`` and exposes intent-named methods that
take/return pure domain models. Ingestion is idempotent: inserts use Postgres
``ON CONFLICT`` against the natural-key constraints declared in ``tables.py``,
so re-fetching the same data never duplicates a row. The caller owns the
transaction (commit/rollback) so multiple repository calls compose atomically.
"""

from collections.abc import Sequence
from datetime import UTC, datetime

from hodlin_contracts import EvidenceRef
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from hodlin_recommend.domain.models import Anomaly, Asset, Explanation, NewsItem, PriceBar
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
        read the rolling z-score consumes. Served by the natural-key index."""
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

    async def recent_for_symbol(
        self, symbol: str, *, since: datetime, until: datetime, limit: int
    ) -> list[NewsItem]:
        """The newest ``limit`` items for a symbol published in [since, until)
        — the "top news" an explanation considers. The upper bound matters when
        explaining historical anomalies: without it, articles *reporting* the
        move would be offered as its cause. Served by ix_news_items_asset."""
        stmt = (
            select(tables.NewsItem)
            .join(tables.Asset, tables.NewsItem.asset_id == tables.Asset.id)
            .where(
                tables.Asset.symbol == symbol,
                tables.NewsItem.published_at >= since,
                tables.NewsItem.published_at < until,
            )
            .order_by(tables.NewsItem.published_at.desc())
            .limit(limit)
        )
        rows = (await self._session.scalars(stmt)).all()
        return [
            NewsItem(
                symbol=symbol,
                source=row.source,
                external_id=row.external_id,
                headline=row.headline,
                url=row.url,
                summary=row.summary,
                published_at=row.published_at,
            )
            for row in rows
        ]


def _anomaly_with_explanation(
    row: tables.Anomaly, explanation_row: tables.Explanation, symbol: str
) -> tuple[Anomaly, Explanation]:
    """Map one joined (anomaly, explanation) row pair to domain models —
    shared by the delivery-queue and latest-explained reads."""
    anomaly = Anomaly(
        symbol=symbol,
        interval=row.interval,
        bar_ts=row.bar_ts,
        z_score=row.z_score,
        return_pct=row.return_pct,
        direction=row.direction,
        window=row.window,
    )
    explanation = Explanation(
        symbol=symbol,
        interval=row.interval,
        bar_ts=row.bar_ts,
        reasoning=explanation_row.reasoning,
        evidence=tuple(EvidenceRef.model_validate(ref) for ref in explanation_row.evidence),
        model_version=explanation_row.model_version,
    )
    return anomaly, explanation


class AnomalyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_many(self, anomalies: Sequence[Anomaly]) -> int:
        """Insert detections, skipping any bar that already tripped — the
        ``(asset_id, interval, bar_ts)`` key means re-running detection over
        the same history never duplicates an anomaly. Returns rows inserted."""
        if not anomalies:
            return 0
        ids = await _resolve_asset_ids(self._session, {a.symbol for a in anomalies})
        rows = [
            {
                "asset_id": ids[anomaly.symbol],
                "interval": anomaly.interval,
                "bar_ts": anomaly.bar_ts,
                "z_score": anomaly.z_score,
                "return_pct": anomaly.return_pct,
                "direction": anomaly.direction,
                "window": anomaly.window,
            }
            for anomaly in anomalies
        ]
        stmt = (
            insert(tables.Anomaly)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_anomalies_natural")
            .returning(tables.Anomaly.id)
        )
        inserted = (await self._session.scalars(stmt)).all()
        return len(inserted)

    async def for_symbol(self, symbol: str, interval: str) -> list[Anomaly]:
        """All detections for a symbol/interval, oldest first."""
        stmt = (
            select(tables.Anomaly)
            .join(tables.Asset, tables.Anomaly.asset_id == tables.Asset.id)
            .where(tables.Asset.symbol == symbol, tables.Anomaly.interval == interval)
            .order_by(tables.Anomaly.bar_ts)
        )
        rows = (await self._session.scalars(stmt)).all()
        return [
            Anomaly(
                symbol=symbol,
                interval=row.interval,
                bar_ts=row.bar_ts,
                z_score=row.z_score,
                return_pct=row.return_pct,
                direction=row.direction,
                window=row.window,
            )
            for row in rows
        ]

    async def explained_unnotified(self, *, limit: int) -> list[tuple[Anomaly, Explanation]]:
        """The delivery queue: anomalies that have their "why" but haven't been
        sent yet, oldest first so alerts arrive in chronological order."""
        stmt = (
            select(tables.Anomaly, tables.Explanation, tables.Asset.symbol)
            .join(tables.Asset, tables.Anomaly.asset_id == tables.Asset.id)
            .join(tables.Explanation, tables.Explanation.anomaly_id == tables.Anomaly.id)
            .where(tables.Anomaly.notified_at.is_(None))
            .order_by(tables.Anomaly.bar_ts, tables.Anomaly.id)
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).all()
        return [
            _anomaly_with_explanation(row, explanation, symbol) for row, explanation, symbol in rows
        ]

    async def latest_explained(self) -> tuple[Anomaly, Explanation] | None:
        """The newest anomaly that has an explanation — the poller's reply to
        an allowlisted 'what's up?', notified or not."""
        stmt = (
            select(tables.Anomaly, tables.Explanation, tables.Asset.symbol)
            .join(tables.Asset, tables.Anomaly.asset_id == tables.Asset.id)
            .join(tables.Explanation, tables.Explanation.anomaly_id == tables.Anomaly.id)
            .order_by(tables.Anomaly.bar_ts.desc(), tables.Anomaly.id.desc())
            .limit(1)
        )
        first = (await self._session.execute(stmt)).first()
        if first is None:
            return None
        row, explanation, symbol = first
        return _anomaly_with_explanation(row, explanation, symbol)

    async def mark_notified(self, symbol: str, interval: str, bar_ts: datetime) -> bool:
        """Atomically claim one anomaly for delivery: flip ``notified_at`` only
        if it is still NULL. Returns whether *this* caller won the claim — the
        compare-and-set that makes "each anomaly notifies once" hold even
        across overlapping processes, not just ticks."""
        asset_id = select(tables.Asset.id).where(tables.Asset.symbol == symbol).scalar_subquery()
        stmt = (
            update(tables.Anomaly)
            .where(
                tables.Anomaly.asset_id == asset_id,
                tables.Anomaly.interval == interval,
                tables.Anomaly.bar_ts == bar_ts,
                tables.Anomaly.notified_at.is_(None),
            )
            .values(notified_at=datetime.now(UTC))
            .returning(tables.Anomaly.id)
        )
        claimed = (await self._session.scalars(stmt)).all()
        return len(claimed) == 1

    async def unexplained(self, *, limit: int) -> list[Anomaly]:
        """The newest ``limit`` anomalies with no explanation yet — the explain
        job's work queue. Newest first because fresh moves are what delivery
        (T9) cares about; the backlog still drains across ticks as each batch
        gets explained and drops out of this query."""
        stmt = (
            select(tables.Anomaly, tables.Asset.symbol)
            .join(tables.Asset, tables.Anomaly.asset_id == tables.Asset.id)
            .join(
                tables.Explanation,
                tables.Explanation.anomaly_id == tables.Anomaly.id,
                isouter=True,
            )
            .where(tables.Explanation.id.is_(None))
            # id breaks ties when assets share a bar_ts — deterministic batches.
            .order_by(tables.Anomaly.bar_ts.desc(), tables.Anomaly.id.desc())
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).all()
        return [
            Anomaly(
                symbol=symbol,
                interval=row.interval,
                bar_ts=row.bar_ts,
                z_score=row.z_score,
                return_pct=row.return_pct,
                direction=row.direction,
                window=row.window,
            )
            for row, symbol in rows
        ]


class UnknownAnomaly(Exception):
    """Raised when an explanation references an anomaly with no stored row.
    Explanations are always minted *for* a detected anomaly, never freestanding."""

    def __init__(self, symbol: str, interval: str, bar_ts: datetime) -> None:
        super().__init__(f"no anomaly row for {symbol!r} {interval} @ {bar_ts.isoformat()}")


class ExplanationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _anomaly_id(self, symbol: str, interval: str, bar_ts: datetime) -> int:
        stmt = (
            select(tables.Anomaly.id)
            .join(tables.Asset, tables.Anomaly.asset_id == tables.Asset.id)
            .where(
                tables.Asset.symbol == symbol,
                tables.Anomaly.interval == interval,
                tables.Anomaly.bar_ts == bar_ts,
            )
        )
        anomaly_id: int | None = await self._session.scalar(stmt)
        if anomaly_id is None:
            raise UnknownAnomaly(symbol, interval, bar_ts)
        return anomaly_id

    async def upsert(self, explanation: Explanation) -> bool:
        """Insert unless the anomaly is already explained (one explanation per
        anomaly, ``uq_explanations_anomaly``). Returns whether a row was
        inserted — re-explaining is a no-op, not an overwrite."""
        anomaly_id = await self._anomaly_id(
            explanation.symbol, explanation.interval, explanation.bar_ts
        )
        stmt = (
            insert(tables.Explanation)
            .values(
                anomaly_id=anomaly_id,
                reasoning=explanation.reasoning,
                # JSON-mode dump: datetimes become ISO strings, JSONB-safe.
                evidence=[ref.model_dump(mode="json") for ref in explanation.evidence],
                model_version=explanation.model_version,
            )
            .on_conflict_do_nothing(constraint="uq_explanations_anomaly")
            .returning(tables.Explanation.id)
        )
        inserted = (await self._session.scalars(stmt)).all()
        return len(inserted) == 1

    async def for_anomaly(self, symbol: str, interval: str, bar_ts: datetime) -> Explanation | None:
        anomaly_id = await self._anomaly_id(symbol, interval, bar_ts)
        stmt = select(tables.Explanation).where(tables.Explanation.anomaly_id == anomaly_id)
        row = await self._session.scalar(stmt)
        if row is None:
            return None
        return Explanation(
            symbol=symbol,
            interval=interval,
            bar_ts=bar_ts,
            reasoning=row.reasoning,
            evidence=tuple(EvidenceRef.model_validate(ref) for ref in row.evidence),
            model_version=row.model_version,
        )


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
