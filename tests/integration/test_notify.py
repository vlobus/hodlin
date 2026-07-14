"""The T9 acceptance against a real Postgres: the allowlisted chat receives a
formatted anomaly+why message, each anomaly notifies exactly once (claim ->
send -> commit on ``anomalies.notified_at``), a failed send releases the
claim for the next tick, and an anomaly without its "why" is never sent.
"""

from datetime import UTC, datetime
from decimal import Decimal

from hodlin_contracts import EvidenceRef
from hodlin_recommend.connectors.base import SourceUnavailable
from hodlin_recommend.domain.models import Anomaly, Asset, Explanation
from hodlin_recommend.ingest import jobs
from hodlin_recommend.store import tables
from hodlin_recommend.store.db import SessionFactory, create_session_factory
from hodlin_recommend.store.repositories import (
    AnomalyRepository,
    AssetRepository,
    ExplanationRepository,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

_CHAT_ID = 42
_BAR_TS = datetime(2024, 6, 24, tzinfo=UTC)


def _anomaly(bar_ts: datetime = _BAR_TS) -> Anomaly:
    return Anomaly(
        symbol="BTC-USD",
        interval="1d",
        bar_ts=bar_ts,
        z_score=Decimal("-2.8"),
        return_pct=Decimal("-4.5271"),
        direction="down",
        window=15,
    )


def _explanation(bar_ts: datetime = _BAR_TS) -> Explanation:
    return Explanation(
        symbol="BTC-USD",
        interval="1d",
        bar_ts=bar_ts,
        reasoning="Likely the exchange hack.",
        evidence=(
            EvidenceRef(
                kind="anomaly", source="hodlin/anomaly", ref="BTC-USD/1d/x", observed_at=bar_ts
            ),
        ),
        model_version="mock:1",
    )


class FakeMessenger:
    def __init__(self, *, dead: bool = False) -> None:
        self.dead = dead
        self.sent: list[tuple[int, str]] = []

    async def send(self, chat_id: int, text: str) -> None:
        if self.dead:
            raise SourceUnavailable("telegram", "api overloaded")
        self.sent.append((chat_id, text))


async def _seed(factory: SessionFactory, *, explained: bool = True) -> None:
    async with factory() as session:
        await AssetRepository(session).upsert(Asset(symbol="BTC-USD", kind="crypto"))
        await AnomalyRepository(session).upsert_many([_anomaly()])
        if explained:
            await ExplanationRepository(session).upsert(_explanation())
        await session.commit()


async def test_allowlisted_chat_gets_anomaly_plus_why_exactly_once(engine: AsyncEngine) -> None:
    factory = create_session_factory(engine)
    await _seed(factory)
    messenger = FakeMessenger()

    first = await jobs.notify_anomalies(factory, messenger=messenger, chat_id=_CHAT_ID)
    second = await jobs.notify_anomalies(factory, messenger=messenger, chat_id=_CHAT_ID)

    assert (first.status, first.items) == ("ok", 1)
    assert (second.status, second.items) == ("ok", 0)
    assert len(messenger.sent) == 1  # once means once, across ticks
    chat_id, text = messenger.sent[0]
    assert chat_id == _CHAT_ID
    assert "BTC-USD" in text
    assert "-4.5271%" in text
    assert "Likely the exchange hack." in text  # the why rides along
    async with factory() as session:
        (run, _) = sorted(
            (await session.scalars(select(tables.IngestRun))).all(), key=lambda r: r.id
        )
        assert (run.job, run.status, run.items) == ("notify_anomalies", "ok", 1)


async def test_failed_send_releases_the_claim_for_the_next_tick(engine: AsyncEngine) -> None:
    factory = create_session_factory(engine)
    await _seed(factory)

    down = await jobs.notify_anomalies(
        factory, messenger=FakeMessenger(dead=True), chat_id=_CHAT_ID
    )
    assert down.status == "error"
    assert down.items == 0

    recovered = FakeMessenger()
    retry = await jobs.notify_anomalies(factory, messenger=recovered, chat_id=_CHAT_ID)

    assert (retry.status, retry.items) == ("ok", 1)  # the claim was rolled back
    assert len(recovered.sent) == 1


async def test_unexplained_anomaly_is_never_sent(engine: AsyncEngine) -> None:
    factory = create_session_factory(engine)
    await _seed(factory, explained=False)
    messenger = FakeMessenger()

    outcome = await jobs.notify_anomalies(factory, messenger=messenger, chat_id=_CHAT_ID)

    assert (outcome.status, outcome.items) == ("ok", 0)
    assert messenger.sent == []  # an alert without its why is not an alert
