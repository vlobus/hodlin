"""Sentiment serving tests — all offline; torch/FinBERT never load.

Covers the T6 acceptance: POST /v1/sentiment returns label + probs +
model_version, and an event-gated concurrency test proves inference runs off
the event loop (a liveness ping is served while an inference is provably in
flight). A real-FinBERT smoke test exists but is opt-in via
HODLIN_TEST_FINBERT=1 (first run downloads ~440 MB).
"""

import asyncio
import os
import sys
import threading
from decimal import Decimal

import httpx
import pytest
from hodlin_recommend.domain.sentiment import SentimentModel, SentimentScore, to_score
from hodlin_recommend.serving.app import create_app


class FakeSentimentModel:
    """Deterministic stand-in: fixed probabilities, instant."""

    model_version = "fake:1"

    def score(self, text: str) -> SentimentScore:
        return to_score({"positive": 0.7, "negative": 0.1, "neutral": 0.2}, self.model_version)


class BlockingSentimentModel(FakeSentimentModel):
    """Blocks its worker thread until the test says otherwise — a forward pass
    of controllable duration, with no wall-clock guessing."""

    def __init__(self) -> None:
        self.started = threading.Event()  # set when inference begins
        self.release = threading.Event()  # test sets this to let inference finish

    def score(self, text: str) -> SentimentScore:
        self.started.set()
        assert self.release.wait(timeout=5), "test never released the inference"
        return super().score(text)


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


def test_to_score_rejects_out_of_range_probabilities() -> None:
    # Raw logits sneaking past softmax must fail loudly, not score "validly".
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        to_score({"positive": 3.2, "negative": -1.1, "neutral": 0.4}, "m:1")


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
    if not os.getenv("HODLIN_TEST_FINBERT"):
        # The default suite must exercise serving without the heavyweights.
        assert "torch" not in sys.modules
        assert "transformers" not in sys.modules


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
    """Start an inference that blocks its worker thread indefinitely, then
    prove the event loop still serves a liveness ping *while it is provably
    in flight*. Event-gated, not timed — no flaky wall-clock assertions. If
    score() ran on the loop, the loop would be stuck inside it and the ping
    (and the release) could never happen."""
    model = BlockingSentimentModel()
    async with _client(model) as client:
        slow = asyncio.create_task(client.post("/v1/sentiment", json={"text": "Profits fell."}))
        # Wait (off the loop) until the forward pass has actually started.
        assert await asyncio.to_thread(model.started.wait, 5)

        ping = await client.get("/health/live")

        assert ping.status_code == 200
        assert not slow.done()  # inference is still grinding in its thread

        model.release.set()
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
