"""The T7 acceptance test: a mocked LLM turns a stored anomaly into a stored
``Explanation`` linked to that anomaly, with >= 1 ``EvidenceRef``. Also proves
the flow is idempotent (an explained anomaly costs no second LLM call) and
that a contract-violating reply stores nothing.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from hodlin_recommend.domain.explanation import MalformedReply
from hodlin_recommend.domain.models import Anomaly, Asset, NewsItem
from hodlin_recommend.domain.sentiment import SentimentScore, to_score
from hodlin_recommend.ingest.explain import explain_anomaly
from hodlin_recommend.store.repositories import (
    AnomalyRepository,
    AssetRepository,
    ExplanationRepository,
    NewsRepository,
    UnknownAnomaly,
)
from sqlalchemy.ext.asyncio import AsyncSession

_BAR_TS = datetime(2024, 6, 24, tzinfo=UTC)

_ANOMALY = Anomaly(
    symbol="BTC-USD",
    interval="1d",
    bar_ts=_BAR_TS,
    z_score=Decimal("-2.801498"),
    return_pct=Decimal("-4.5271"),
    direction="down",
    window=15,
)

_GOOD_REPLY = '{"reasoning": "Most likely the exchange hack.", "evidence_indices": [0]}'


class MockLLM:
    model_version = "mock:1"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


class FakeSentimentModel:
    model_version = "fake:1"

    def score(self, text: str) -> SentimentScore:
        return to_score({"positive": 0.1, "negative": 0.8, "neutral": 0.1}, self.model_version)


async def _seed(session: AsyncSession) -> None:
    await AssetRepository(session).upsert(Asset(symbol="BTC-USD", kind="crypto"))
    await AnomalyRepository(session).upsert_many([_ANOMALY])
    await NewsRepository(session).upsert_many(
        [
            NewsItem(
                symbol="BTC-USD",
                source="finnhub",
                external_id="n-1",
                headline="Major exchange hacked, funds drained",
                url="https://example.test/hack",
                published_at=datetime(2024, 6, 23, 18, 0, tzinfo=UTC),
            ),
            NewsItem(
                symbol="BTC-USD",
                source="finnhub",
                external_id="n-2",
                headline="Unrelated: new logo announced",
                published_at=datetime(2024, 6, 22, 9, 0, tzinfo=UTC),
            ),
        ]
    )
    await session.commit()


async def test_mocked_llm_yields_stored_linked_explanation(session: AsyncSession) -> None:
    await _seed(session)
    llm = MockLLM(_GOOD_REPLY)

    explanation = await explain_anomaly(
        session, _ANOMALY, llm=llm, sentiment_model=FakeSentimentModel()
    )

    assert explanation is not None
    assert llm.calls == 1
    stored = await ExplanationRepository(session).for_anomaly("BTC-USD", "1d", _BAR_TS)
    assert stored is not None  # linked to its anomaly by natural key
    assert stored.reasoning == "Most likely the exchange hack."
    assert stored.model_version == "mock:1"
    assert len(stored.evidence) >= 1
    kinds = [ref.kind for ref in stored.evidence]
    assert kinds == ["anomaly", "news", "sentiment"]  # cited [0] only, not n-2
    assert stored.evidence[1].ref == "https://example.test/hack"


async def test_explained_anomaly_is_not_reexplained(session: AsyncSession) -> None:
    await _seed(session)
    llm = MockLLM(_GOOD_REPLY)

    first = await explain_anomaly(session, _ANOMALY, llm=llm, sentiment_model=FakeSentimentModel())
    second = await explain_anomaly(session, _ANOMALY, llm=llm, sentiment_model=FakeSentimentModel())

    assert first is not None
    assert second is None
    assert llm.calls == 1  # idempotency saves the money, not just the row


async def test_malformed_reply_stores_nothing(session: AsyncSession) -> None:
    await _seed(session)
    llm = MockLLM("I think it was the hack. Definitely sell everything.")

    with pytest.raises(MalformedReply):
        await explain_anomaly(session, _ANOMALY, llm=llm, sentiment_model=FakeSentimentModel())

    assert await ExplanationRepository(session).for_anomaly("BTC-USD", "1d", _BAR_TS) is None


async def test_explanation_for_unknown_anomaly_raises(session: AsyncSession) -> None:
    with pytest.raises(UnknownAnomaly):
        await ExplanationRepository(session).for_anomaly("BTC-USD", "1d", _BAR_TS)
