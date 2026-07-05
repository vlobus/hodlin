"""Sentiment serving tests — all offline; torch/FinBERT never load.

Covers the T6 acceptance: POST /v1/sentiment returns label + probs +
model_version, and a concurrency test proves inference runs off the event
loop (a liveness ping stays fast while a slow inference is in flight).
A real-FinBERT smoke test exists but is opt-in via HODLIN_TEST_FINBERT=1
(first run downloads ~440 MB).
"""

import asyncio
import os
import time
from decimal import Decimal

import httpx
import pytest
from hodlin_recommend.domain.sentiment import SentimentModel, SentimentScore, to_score
from hodlin_recommend.serving.app import create_app

_SLOW_INFERENCE = 0.3  # seconds the fake model blocks its thread
_PING_BUDGET = 0.15  # a ping must return well inside the inference window


class FakeSentimentModel:
    """Deterministic stand-in: fixed probabilities, optional thread-blocking
    delay to simulate a real forward pass."""

    model_version = "fake:1"

    def __init__(self, delay: float = 0.0) -> None:
        self._delay = delay

    def score(self, text: str) -> SentimentScore:
        if self._delay:
            time.sleep(self._delay)  # blocks its thread, as real inference would
        return to_score({"positive": 0.7, "negative": 0.1, "neutral": 0.2}, self.model_version)


def _client(model: SentimentModel) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_app(sentiment_model=model))
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# Pure mapping --------------------------------------------------------------


def test_to_score_picks_argmax_and_quantizes() -> None:
    score = to_score({"positive": 0.1, "negative": 0.85, "neutral": 0.05}, "m:1")

    assert score.label == "negative"
    assert score.prob_negative == Decimal("0.850000")
    assert score.prob_positive == Decimal("0.100000")
    assert score.model_version == "m:1"


def test_to_score_rejects_wrong_label_set() -> None:
    with pytest.raises(ValueError, match="expected probabilities"):
        to_score({"positive": 0.5, "negative": 0.5}, "m:1")


def test_fake_satisfies_the_protocol() -> None:
    assert isinstance(FakeSentimentModel(), SentimentModel)


# Endpoint ------------------------------------------------------------------


async def test_sentiment_endpoint_returns_label_probs_and_model_version() -> None:
    async with _client(FakeSentimentModel()) as client:
        response = await client.post("/v1/sentiment", json={"text": "Shares surged 12%."})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    # Decimals cross the wire as strings — exact, never a binary float.
    assert body["probs"] == {"positive": "0.700000", "negative": "0.100000", "neutral": "0.200000"}
    assert body["model_version"] == "fake:1"


async def test_sentiment_endpoint_rejects_blank_text() -> None:
    async with _client(FakeSentimentModel()) as client:
        assert (await client.post("/v1/sentiment", json={"text": "   "})).status_code == 422
        assert (await client.post("/v1/sentiment", json={})).status_code == 422


async def test_health_live() -> None:
    async with _client(FakeSentimentModel()) as client:
        response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Concurrency: the T6 acceptance -------------------------------------------


async def test_inference_does_not_block_the_event_loop() -> None:
    """Fire a slow inference, then prove the loop still serves a liveness ping
    while that inference is grinding in its worker thread. If score() ran on
    the loop, the ping couldn't complete before the inference finishes."""
    async with _client(FakeSentimentModel(delay=_SLOW_INFERENCE)) as client:
        slow = asyncio.create_task(
            client.post("/v1/sentiment", json={"text": "Profits collapsed."})
        )
        await asyncio.sleep(0.05)  # let the slow request reach the model

        started = time.monotonic()
        ping = await client.get("/health/live")
        ping_latency = time.monotonic() - started

        assert ping.status_code == 200
        assert not slow.done()  # inference genuinely still in flight
        assert ping_latency < _PING_BUDGET

        response = await slow
        assert response.status_code == 200
        assert response.json()["label"] == "positive"


# Real model (opt-in) --------------------------------------------------------


@pytest.mark.skipif(
    not os.getenv("HODLIN_TEST_FINBERT"),
    reason="set HODLIN_TEST_FINBERT=1 to run real FinBERT (downloads ~440 MB on first run)",
)
async def test_real_finbert_scores_obvious_sentiment() -> None:
    from hodlin_recommend.domain.sentiment import FinBertModel

    model = FinBertModel()
    async with _client(model) as client:
        good = await client.post(
            "/v1/sentiment", json={"text": "The company beat expectations and raised guidance."}
        )
        bad = await client.post(
            "/v1/sentiment",
            json={"text": "The company missed estimates and slashed its dividend."},
        )

    assert good.json()["label"] == "positive"
    assert bad.json()["label"] == "negative"
    assert good.json()["model_version"] == "finbert:ProsusAI/finbert"
