"""Scheduler wiring tests — offline, no Postgres, no real ticks on real time.

What's asserted is *configuration and lifecycle*, the parts that are ours:
the four recurring jobs exist with overlap protection (``max_instances=1``),
coalescing, and a misfire grace; backfill registers as a one-shot; the app
lifespan starts the scheduler, a job actually fires on the loop, and shutdown
leaves nothing running. Firing on a *schedule* against a real database is the
integration suite's job (``tests/integration/test_jobs.py``).
"""

import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from hodlin_recommend.domain.models import NewsItem, PriceBar
from hodlin_recommend.domain.sentiment import SentimentScore, to_score
from hodlin_recommend.ingest.scheduler import (
    BARS_EVERY_S,
    DETECT_EVERY_S,
    EXPLAIN_EVERY_S,
    MISFIRE_GRACE_S,
    NEWS_EVERY_S,
    build_scheduler,
)
from hodlin_recommend.serving.app import SchedulerLike, create_app
from hodlin_recommend.store.db import create_engine, create_session_factory


class FakeSentimentModel:
    model_version = "fake:1"

    def score(self, text: str) -> SentimentScore:
        return to_score({"positive": 0.1, "negative": 0.8, "neutral": 0.1}, self.model_version)


class FakeNewsSource:
    source = "fake-news"

    async def get_news(self, symbol: str, since: datetime) -> list[NewsItem]:
        return []

    async def health(self) -> bool:
        return True


class FakeBarSource:
    source = "fake-bars"

    async def get_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[PriceBar]:
        return []

    async def health(self) -> bool:
        return True


class MockLLM:
    model_version = "mock:1"

    async def complete(self, *, system: str, user: str) -> str:
        return '{"reasoning": "why", "evidence_indices": []}'


def _build(*, backfill_on_start: bool = True) -> AsyncIOScheduler:
    # A real factory over an engine that never connects — jobs never run here.
    factory = create_session_factory(create_engine("postgresql+asyncpg://x:x@localhost/x"))
    return build_scheduler(
        session_factory=factory,
        bar_source=FakeBarSource(),
        news_source=FakeNewsSource(),
        llm=MockLLM(),
        sentiment_model=FakeSentimentModel(),
        backfill_on_start=backfill_on_start,
    )


EXPECTED_INTERVALS = {
    "ingest_bars": BARS_EVERY_S,
    "ingest_news": NEWS_EVERY_S,
    "detect_anomalies": DETECT_EVERY_S,
    "explain_anomalies": EXPLAIN_EVERY_S,
}


async def test_four_recurring_jobs_with_overlap_protection() -> None:
    scheduler = _build()
    scheduler.start(paused=True)  # materializes pending jobs without firing any
    try:
        jobs = {job.id: job for job in scheduler.get_jobs()}
        assert set(jobs) == {*EXPECTED_INTERVALS, "backfill"}
        for job_id, every in EXPECTED_INTERVALS.items():
            job = jobs[job_id]
            assert isinstance(job.trigger, IntervalTrigger)
            assert job.trigger.interval.total_seconds() == every
            assert job.max_instances == 1  # a slow run means a skipped tick, not overlap
            assert job.coalesce is True
            assert job.misfire_grace_time == MISFIRE_GRACE_S
        assert isinstance(jobs["backfill"].trigger, DateTrigger)  # one-shot at startup
    finally:
        scheduler.shutdown(wait=False)


async def test_backfill_can_be_left_out() -> None:
    scheduler = _build(backfill_on_start=False)
    scheduler.start(paused=True)
    try:
        assert {job.id for job in scheduler.get_jobs()} == set(EXPECTED_INTERVALS)
    finally:
        scheduler.shutdown(wait=False)


def test_asyncio_scheduler_satisfies_the_protocol() -> None:
    assert isinstance(AsyncIOScheduler(timezone=UTC), SchedulerLike)


async def test_lifespan_starts_scheduler_fires_jobs_and_stops_clean() -> None:
    """The wiring end to end, without a database: entering the lifespan starts
    the scheduler, a registered job runs on the app's own loop, and leaving
    the lifespan stops the scheduler and closes the resource stack — in that
    order, so a job never outlives its resources."""
    fired = asyncio.Event()

    async def tick() -> None:
        fired.set()

    scheduler = AsyncIOScheduler(timezone=UTC)
    scheduler.add_job(tick, "interval", seconds=0.05, id="tick")

    closed = asyncio.Event()

    async def close_resources() -> None:
        assert not scheduler.running  # scheduler must stop before resources go
        closed.set()

    resources = AsyncExitStack()
    resources.push_async_callback(close_resources)

    app = create_app(sentiment_model=FakeSentimentModel(), scheduler=scheduler, resources=resources)
    async with app.router.lifespan_context(app):
        assert scheduler.running
        await asyncio.wait_for(fired.wait(), timeout=5)

    assert not scheduler.running
    assert closed.is_set()


async def test_ready_is_503_when_nothing_is_wired() -> None:
    """A serving-only app (no DB, no scheduler) is alive but not ready."""
    app = create_app(sentiment_model=FakeSentimentModel())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        live = await client.get("/health/live")
        ready = await client.get("/health/ready")

    assert live.status_code == 200
    assert ready.status_code == 503
    assert "no database configured" in ready.json()["problems"]
