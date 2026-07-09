"""The T8 acceptance against a real Postgres: every job run leaves an
``ingest_runs`` audit row (ok and error alike), jobs are idempotent across
ticks, and a scheduler started by the app lifespan fires a job whose run is
visible in the table while ``/health/ready`` answers 200 — then shuts down
clean. The LLM stays mocked (T7 rule) and sources are fakes: what's under
test is orchestration and audit, not providers.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
from hodlin_recommend.connectors.base import SourceUnavailable
from hodlin_recommend.domain.asset_config import AssetConfig
from hodlin_recommend.domain.explanation import LLMUnavailable
from hodlin_recommend.domain.models import Anomaly, Asset, NewsItem, PriceBar
from hodlin_recommend.domain.sentiment import SentimentScore, to_score
from hodlin_recommend.ingest import jobs
from hodlin_recommend.ingest.jobs import SessionFactory
from hodlin_recommend.ingest.scheduler import build_scheduler
from hodlin_recommend.serving.app import create_app
from hodlin_recommend.store import tables
from hodlin_recommend.store.db import create_session_factory
from hodlin_recommend.store.repositories import (
    AnomalyRepository,
    AssetRepository,
    ExplanationRepository,
    NewsRepository,
    PriceBarRepository,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

_ASSET = AssetConfig(symbol="BTC-USD", kind="crypto", name="Bitcoin / USD", window=5)
_BAR_TS = datetime(2024, 6, 24, tzinfo=UTC)

# A quiet tail and one violent last bar: the +11% close against a ~0.4%-sigma
# baseline trips any sane threshold at window=5.
_CLOSES = ["100", "100.5", "100.2", "100.7", "100.4", "100.6", "112"]


def _bars() -> list[PriceBar]:
    return [
        PriceBar(
            symbol=_ASSET.symbol,
            interval=_ASSET.interval,
            ts=_BAR_TS - timedelta(days=len(_CLOSES) - 1 - i),
            open=Decimal(close),
            high=Decimal(close),
            low=Decimal(close),
            close=Decimal(close),
            source="fake-bars",
        )
        for i, close in enumerate(_CLOSES)
    ]


class FakeBarSource:
    source = "fake-bars"

    def __init__(self, bars: list[PriceBar] | None = None, boom: Exception | None = None) -> None:
        self.bars = bars or []
        self.boom = boom

    async def get_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[PriceBar]:
        if self.boom is not None:
            raise self.boom
        return self.bars

    async def health(self) -> bool:
        return self.boom is None


class FakeNewsSource:
    source = "fake-news"

    def __init__(self, items: list[NewsItem] | None = None) -> None:
        self.items = items or []

    async def get_news(self, symbol: str, since: datetime) -> list[NewsItem]:
        return self.items

    async def health(self) -> bool:
        return True


class MockLLM:
    model_version = "mock:1"

    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.calls = 0

    async def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class FakeSentimentModel:
    model_version = "fake:1"

    def score(self, text: str) -> SentimentScore:
        return to_score({"positive": 0.1, "negative": 0.8, "neutral": 0.1}, self.model_version)


async def _runs(session: AsyncSession, job: str) -> list[tables.IngestRun]:
    stmt = select(tables.IngestRun).where(tables.IngestRun.job == job)
    return list((await session.scalars(stmt)).all())


# Audit rows -----------------------------------------------------------------


async def test_ingest_bars_job_stores_bars_and_audits_ok(engine: AsyncEngine) -> None:
    factory = create_session_factory(engine)
    outcome = await jobs.ingest_bars(factory, FakeBarSource(_bars()), [_ASSET])

    assert outcome.status == "ok"
    assert outcome.items == len(_CLOSES)
    async with factory() as session:
        (run,) = await _runs(session, "ingest_bars")
        assert run.status == "ok"
        assert run.items == len(_CLOSES)
        assert run.finished_at is not None
        stored = await PriceBarRepository(session).recent(_ASSET.symbol, _ASSET.interval, 10)
        assert len(stored) == len(_CLOSES)


async def test_dead_source_is_a_skip_note_not_a_failed_run(engine: AsyncEngine) -> None:
    factory = create_session_factory(engine)
    source = FakeBarSource(boom=SourceUnavailable("fake-bars", "connection refused"))
    outcome = await jobs.ingest_bars(factory, source, [_ASSET])

    assert outcome.status == "ok"  # the job ran fine; the *source* is down
    assert outcome.items == 0
    assert outcome.detail is not None and _ASSET.symbol in outcome.detail


async def test_crashed_job_leaves_an_error_run_and_no_data(engine: AsyncEngine) -> None:
    factory = create_session_factory(engine)
    outcome = await jobs.ingest_bars(factory, FakeBarSource(boom=RuntimeError("boom")), [_ASSET])

    assert outcome.status == "error"
    async with factory() as session:
        (run,) = await _runs(session, "ingest_bars")
        assert run.status == "error"
        assert run.detail == "RuntimeError: boom"
        assert run.finished_at is not None  # the row closes even on a crash
        assert await PriceBarRepository(session).recent(_ASSET.symbol, _ASSET.interval, 10) == []


# Detection + explanation ticks ----------------------------------------------


async def test_detect_job_flags_the_spike_and_reruns_idempotently(engine: AsyncEngine) -> None:
    factory = create_session_factory(engine)
    await jobs.ingest_bars(factory, FakeBarSource(_bars()), [_ASSET])

    first = await jobs.detect_anomalies(factory, [_ASSET])
    second = await jobs.detect_anomalies(factory, [_ASSET])

    assert first.items == 1
    assert second.items == 0  # same bars, same anomaly, zero new rows
    async with factory() as session:
        (anomaly,) = await AnomalyRepository(session).for_symbol(_ASSET.symbol, _ASSET.interval)
        assert anomaly.direction == "up"
        assert anomaly.bar_ts == _BAR_TS


async def _seed_anomaly(factory: SessionFactory, with_news: bool = True) -> None:
    async with factory() as session:
        await AssetRepository(session).upsert(Asset(symbol=_ASSET.symbol, kind=_ASSET.kind))
        await AnomalyRepository(session).upsert_many(
            [
                Anomaly(
                    symbol=_ASSET.symbol,
                    interval=_ASSET.interval,
                    bar_ts=_BAR_TS,
                    z_score=Decimal("3.1"),
                    return_pct=Decimal("11.3"),
                    direction="up",
                    window=5,
                )
            ]
        )
        if with_news:
            await NewsRepository(session).upsert_many(
                [
                    NewsItem(
                        symbol=_ASSET.symbol,
                        source="fake-news",
                        external_id="n-1",
                        headline="ETF approval rumors swirl",
                        published_at=_BAR_TS - timedelta(hours=6),
                    )
                ]
            )
        await session.commit()


async def test_explain_job_explains_newest_unexplained_then_goes_quiet(
    engine: AsyncEngine,
) -> None:
    factory = create_session_factory(engine)
    await _seed_anomaly(factory)
    llm = MockLLM('{"reasoning": "Likely the ETF rumors.", "evidence_indices": [0]}')

    first = await jobs.explain_anomalies(factory, llm=llm, sentiment_model=FakeSentimentModel())
    second = await jobs.explain_anomalies(factory, llm=llm, sentiment_model=FakeSentimentModel())

    assert (first.status, first.items) == ("ok", 1)
    assert (second.status, second.items) == ("ok", 0)
    assert llm.calls == 1  # the second tick spends nothing
    async with factory() as session:
        stored = await ExplanationRepository(session).for_anomaly(
            _ASSET.symbol, _ASSET.interval, _BAR_TS
        )
        assert stored is not None
        assert stored.reasoning == "Likely the ETF rumors."


async def test_llm_down_marks_the_run_error_and_stores_nothing(engine: AsyncEngine) -> None:
    factory = create_session_factory(engine)
    await _seed_anomaly(factory)
    llm = MockLLM(LLMUnavailable("api overloaded"))

    outcome = await jobs.explain_anomalies(factory, llm=llm, sentiment_model=FakeSentimentModel())

    assert outcome.status == "error"
    assert outcome.items == 0
    async with factory() as session:
        (run,) = await _runs(session, "explain_anomalies")
        assert run.status == "error"
        assert run.detail is not None and "unavailable" in run.detail
        assert (
            await ExplanationRepository(session).for_anomaly(
                _ASSET.symbol, _ASSET.interval, _BAR_TS
            )
            is None
        )


# The acceptance: scheduler fires -> ingest_runs -> ready -> clean stop ------


async def test_scheduled_run_lands_in_ingest_runs_and_app_reports_ready(
    engine: AsyncEngine,
) -> None:
    factory = create_session_factory(engine)
    scheduler = build_scheduler(
        session_factory=factory,
        bar_source=FakeBarSource(_bars()),
        news_source=FakeNewsSource(),
        llm=MockLLM('{"reasoning": "why", "evidence_indices": []}'),
        sentiment_model=FakeSentimentModel(),
        assets=[_ASSET],
    )
    app = create_app(
        sentiment_model=FakeSentimentModel(), scheduler=scheduler, session_factory=factory
    )
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/health/ready")).status_code == 200

        async def backfill_completed() -> bool:
            async with factory() as session:
                runs = await _runs(session, "backfill")
                return any(run.status == "ok" for run in runs)

        # The one-shot backfill fires as soon as the scheduler starts. The
        # signal lives in Postgres, so there's no Event to await — polling
        # the audit table is the point (bounded by wait_for).
        async def until_landed() -> None:
            while not await backfill_completed():  # noqa: ASYNC110
                await asyncio.sleep(0.05)

        await asyncio.wait_for(until_landed(), timeout=10)

    assert not scheduler.running  # lifespan exit really stopped it
