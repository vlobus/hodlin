"""Scheduler wiring (T8, D5): four recurring jobs on an ``AsyncIOScheduler``.

APScheduler's asyncio scheduler runs coroutine jobs as tasks on the app's own
event loop — no extra threads or processes, and the jobs share the process's
single FinBERT/DB/HTTP resources through DI. Overlap protection comes from
``max_instances=1`` (a tick is skipped while the previous run of that job is
still going, and the skip is logged), ``coalesce=True`` collapses a pile of
missed ticks into one run, and ``misfire_grace_time`` bounds how stale a
missed tick may be and still fire.

Intervals are tuning, not secrets (D17): daily bars don't need minute-level
polling; the explain tick is shorter so a fresh anomaly gets its "why" fast.
The builder only assembles — construction of the concretes stays in the
composition root (``main.py``), and tests hand in fakes.
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC
from functools import partial

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from hodlin_recommend.connectors.base import NewsSource, PriceBarSource
from hodlin_recommend.domain.asset_config import DEFAULT_ASSETS, AssetConfig
from hodlin_recommend.domain.explanation import ExplainerLLM
from hodlin_recommend.domain.sentiment import SentimentModel
from hodlin_recommend.ingest import jobs
from hodlin_recommend.store.db import SessionFactory

# Tuning, not secrets (D17).
BARS_EVERY_S = 900
NEWS_EVERY_S = 900
DETECT_EVERY_S = 900
EXPLAIN_EVERY_S = 300
MISFIRE_GRACE_S = 60

JOB_DEFAULTS = {
    "max_instances": 1,  # a job never overlaps itself; a busy tick is skipped
    "coalesce": True,  # N missed ticks -> 1 catch-up run, not N
    "misfire_grace_time": MISFIRE_GRACE_S,
}


def build_scheduler(
    *,
    session_factory: SessionFactory,
    bar_source: PriceBarSource,
    news_source: NewsSource,
    llm: ExplainerLLM,
    sentiment_model: SentimentModel,
    inference_executor: ThreadPoolExecutor | None = None,
    assets: Sequence[AssetConfig] = DEFAULT_ASSETS,
    backfill_on_start: bool = True,
) -> AsyncIOScheduler:
    """Assemble the scheduler; ``start()`` belongs to the app lifespan.

    ``backfill_on_start`` registers the cold-start pass as a one-shot job that
    fires as soon as the scheduler starts — audited in ``ingest_runs`` like
    every recurring tick, and off the startup path so the API is up while
    history fills in.
    """
    scheduler = AsyncIOScheduler(timezone=UTC, job_defaults=JOB_DEFAULTS)
    scheduler.add_job(
        partial(jobs.ingest_bars, session_factory, bar_source, assets),
        "interval",
        seconds=BARS_EVERY_S,
        id="ingest_bars",
    )
    scheduler.add_job(
        partial(jobs.ingest_news, session_factory, news_source, assets),
        "interval",
        seconds=NEWS_EVERY_S,
        id="ingest_news",
    )
    scheduler.add_job(
        partial(jobs.detect_anomalies, session_factory, assets),
        "interval",
        seconds=DETECT_EVERY_S,
        id="detect_anomalies",
    )
    scheduler.add_job(
        partial(
            jobs.explain_anomalies,
            session_factory,
            llm=llm,
            sentiment_model=sentiment_model,
            inference_executor=inference_executor,
        ),
        "interval",
        seconds=EXPLAIN_EVERY_S,
        id="explain_anomalies",
    )
    if backfill_on_start:
        # No trigger = a date trigger of "now" — stamped at *build* time, so
        # the grace must be unlimited: if startup takes longer than the
        # default 60s (model download, slow bind), the cold start must still
        # run late rather than be silently dropped as a misfire.
        scheduler.add_job(
            partial(jobs.run_backfill, session_factory, bar_source, assets),
            id="backfill",
            misfire_grace_time=None,
        )
    return scheduler
