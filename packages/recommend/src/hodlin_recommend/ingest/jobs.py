"""Scheduled jobs (T8): small, audited units of work the scheduler ticks.

Every job runs through ``run_audited``: an ``ingest_runs`` row opens — and
commits — before any work, so an in-flight run is visible in the table, and
closes ok/error with an item count and detail. Errors are recorded, never
raised: one bad tick must not take the scheduler down; the next tick retries.

Jobs are deliberately re-runnable. Every write path upserts against natural
keys, and the fetch lookbacks overlap on purpose — a missed tick heals on the
next one, and idempotency makes the overlap free. Each job reads whatever is
stored rather than depending on another job having just run, so their relative
order within a tick doesn't matter.
"""

from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from hodlin_recommend.connectors.base import NewsSource, PriceBarSource, SourceUnavailable
from hodlin_recommend.domain.anomaly import detect_series
from hodlin_recommend.domain.asset_config import AssetConfig
from hodlin_recommend.domain.explanation import ExplainerLLM, LLMUnavailable, MalformedReply
from hodlin_recommend.domain.models import Asset
from hodlin_recommend.domain.sentiment import SentimentModel
from hodlin_recommend.ingest.backfill import backfill_assets
from hodlin_recommend.ingest.explain import explain_anomaly
from hodlin_recommend.store.db import SessionFactory
from hodlin_recommend.store.repositories import (
    AnomalyRepository,
    AssetRepository,
    IngestRunRepository,
    NewsRepository,
    PriceBarRepository,
)

# Tuning, not secrets (D17). Lookbacks overlap several ticks so gaps heal;
# DETECT_TAIL bounds how many recent bars get (re-)scored per tick; the
# explain batch caps LLM spend per tick.
BARS_LOOKBACK = timedelta(days=5)
NEWS_LOOKBACK = timedelta(days=3)
DETECT_TAIL = 8
EXPLAIN_BATCH = 5


@dataclass(frozen=True)
class JobOutcome:
    """What one job run accomplished — becomes the ``ingest_runs`` row."""

    items: int = 0
    detail: str | None = None
    status: str = "ok"  # "ok" | "error"


async def run_audited(
    session_factory: SessionFactory,
    job: str,
    work: Callable[[AsyncSession], Awaitable[JobOutcome]],
) -> JobOutcome:
    """Run ``work`` inside an ``ingest_runs`` audit row.

    The "running" row commits before work starts, so overlap and hangs are
    observable in the table while they happen. An exception rolls back the
    work, records an error row, and is swallowed — the scheduler must outlive
    any single tick. Two honest limits: if the *database* is down, the audit
    write itself raises out of the job (APScheduler logs it and survives);
    and a tick cancelled at shutdown leaves its row at "running" — a true
    record, redone at the next startup since every job is idempotent.
    """
    async with session_factory() as session:
        runs = IngestRunRepository(session)
        run_id = await runs.start(job)
        await session.commit()
        try:
            outcome = await work(session)
        except Exception as exc:
            await session.rollback()
            outcome = JobOutcome(status="error", detail=f"{type(exc).__name__}: {exc}")
        await runs.finish(run_id, status=outcome.status, items=outcome.items, detail=outcome.detail)
        await session.commit()
        return outcome


async def _ensure_assets(session: AsyncSession, assets: Sequence[AssetConfig]) -> None:
    repo = AssetRepository(session)
    for config in assets:
        await repo.upsert(Asset(symbol=config.symbol, kind=config.kind, name=config.name))


async def ingest_bars(
    session_factory: SessionFactory,
    source: PriceBarSource,
    assets: Sequence[AssetConfig],
) -> JobOutcome:
    """Fetch the recent bar window per asset and upsert. A dead source skips
    that asset with a recorded reason — same policy as backfill."""

    async def work(session: AsyncSession) -> JobOutcome:
        await _ensure_assets(session, assets)
        repo = PriceBarRepository(session)
        now = datetime.now(UTC)
        inserted = 0
        skipped: list[str] = []
        for config in assets:
            try:
                bars = await source.get_candles(
                    config.symbol, config.interval, now - BARS_LOOKBACK, now
                )
            except SourceUnavailable as exc:
                skipped.append(f"{config.symbol}: {exc}")
                continue
            inserted += await repo.upsert_many(bars)
        return JobOutcome(items=inserted, detail="; ".join(skipped) or None)

    return await run_audited(session_factory, "ingest_bars", work)


async def ingest_news(
    session_factory: SessionFactory,
    source: NewsSource,
    assets: Sequence[AssetConfig],
) -> JobOutcome:
    """Fetch recent news per asset and upsert against the natural key."""

    async def work(session: AsyncSession) -> JobOutcome:
        await _ensure_assets(session, assets)
        repo = NewsRepository(session)
        since = datetime.now(UTC) - NEWS_LOOKBACK
        inserted = 0
        skipped: list[str] = []
        for config in assets:
            try:
                items = await source.get_news(config.symbol, since)
            except SourceUnavailable as exc:
                skipped.append(f"{config.symbol}: {exc}")
                continue
            inserted += await repo.upsert_many(items)
        return JobOutcome(items=inserted, detail="; ".join(skipped) or None)

    return await run_audited(session_factory, "ingest_news", work)


async def detect_anomalies(
    session_factory: SessionFactory,
    assets: Sequence[AssetConfig],
) -> JobOutcome:
    """Score the tail of each asset's stored bars; upsert anything that trips.
    Reads only the store — never a live source — so it works the same whether
    the bars arrived via backfill or the ingest tick."""

    async def work(session: AsyncSession) -> JobOutcome:
        bar_repo = PriceBarRepository(session)
        anomaly_repo = AnomalyRepository(session)
        inserted = 0
        for config in assets:
            stored = await bar_repo.recent(
                config.symbol, config.interval, limit=config.window + DETECT_TAIL + 1
            )
            anomalies = detect_series(stored, window=config.window, threshold=config.threshold)
            inserted += await anomaly_repo.upsert_many(anomalies)
        return JobOutcome(items=inserted)

    return await run_audited(session_factory, "detect_anomalies", work)


async def explain_anomalies(
    session_factory: SessionFactory,
    *,
    llm: ExplainerLLM,
    sentiment_model: SentimentModel,
    inference_executor: ThreadPoolExecutor | None = None,
) -> JobOutcome:
    """Explain up to ``EXPLAIN_BATCH`` unexplained anomalies, newest first.

    ``LLMUnavailable`` aborts the batch (the provider is down for all of them;
    the next tick retries) and marks the run "error". ``MalformedReply`` skips
    just that anomaly — a fresh call may still validate next tick.
    """

    async def work(session: AsyncSession) -> JobOutcome:
        anomalies = await AnomalyRepository(session).unexplained(limit=EXPLAIN_BATCH)
        explained = 0
        notes: list[str] = []
        for anomaly in anomalies:
            key = f"{anomaly.symbol}/{anomaly.interval}@{anomaly.bar_ts.isoformat()}"
            try:
                result = await explain_anomaly(
                    session,
                    anomaly,
                    llm=llm,
                    sentiment_model=sentiment_model,
                    inference_executor=inference_executor,
                )
            except LLMUnavailable as exc:
                notes.append(str(exc))
                return JobOutcome(items=explained, detail="; ".join(notes), status="error")
            except MalformedReply as exc:
                notes.append(f"{key}: {exc}")
                continue
            if result is not None:
                explained += 1
        return JobOutcome(items=explained, detail="; ".join(notes) or None)

    return await run_audited(session_factory, "explain_anomalies", work)


async def run_backfill(
    session_factory: SessionFactory,
    source: PriceBarSource,
    assets: Sequence[AssetConfig],
) -> JobOutcome:
    """The cold-start pass (T5), audited like any other job so the startup
    run is visible in ``ingest_runs`` alongside the recurring ticks."""

    async def work(session: AsyncSession) -> JobOutcome:
        results = await backfill_assets(session, source, assets)
        bars = sum(result.bars_inserted for result in results)
        anomalies = sum(result.anomalies_inserted for result in results)
        skipped = [f"{result.symbol}: {result.skipped}" for result in results if result.skipped]
        detail = "; ".join([f"anomalies={anomalies}", *skipped])
        return JobOutcome(items=bars, detail=detail)

    return await run_audited(session_factory, "backfill", work)
