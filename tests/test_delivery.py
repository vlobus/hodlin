"""Delivery tests — all offline (respx mocks the Bot API; no token, no network).

Covers the T9 pieces that are pure or mockable without Postgres: formatting
escapes untrusted text, the thin client speaks the Bot API and degrades to
``SourceUnavailable``, and the poller enforces the single-ID allowlist —
a stranger's message produces no reply at all, while the allowlisted user
gets the injected reply text. The notify-once flow runs against real Postgres
in ``tests/integration/test_notify.py``.
"""

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
import pytest
import respx
from hodlin_contracts import EvidenceRef
from hodlin_recommend.connectors.base import RateLimiter, RetryPolicy, SourceUnavailable
from hodlin_recommend.delivery.formatting import format_anomaly, format_status
from hodlin_recommend.delivery.poller import UpdatePoller
from hodlin_recommend.delivery.telegram import Messenger, TelegramClient, UpdateSource
from hodlin_recommend.domain.models import Anomaly, Explanation

_BASE = "https://api.test"
_SEND_URL = f"{_BASE}/botTOKEN/sendMessage"
_FAST = RetryPolicy(attempts=2, wait_initial=0.0, wait_max=0.0)

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


def _explanation(reasoning: str = "Likely the exchange hack.") -> Explanation:
    return Explanation(
        symbol="BTC-USD",
        interval="1d",
        bar_ts=_BAR_TS,
        reasoning=reasoning,
        evidence=(
            EvidenceRef(
                kind="anomaly",
                source="hodlin/anomaly",
                ref="BTC-USD/1d/x",
                observed_at=_BAR_TS,
            ),
            EvidenceRef(
                kind="news",
                source="finnhub",
                ref="https://example.test/hack",
                observed_at=_BAR_TS,
            ),
        ),
        model_version="anthropic:claude-x",
    )


@pytest.fixture
def rate() -> RateLimiter:
    return RateLimiter(10_000)


# Formatting — the escaping boundary -----------------------------------------


def test_format_carries_the_numbers_and_the_why() -> None:
    text = format_anomaly(_ANOMALY, _explanation())

    assert "<b>BTC-USD</b>" in text
    assert "-4.5271%" in text
    assert "z-score -2.801498" in text
    assert "Likely the exchange hack." in text
    assert "1 news source(s) cited" in text  # the anomaly self-ref doesn't count


def test_format_escapes_untrusted_reasoning() -> None:
    # LLM prose over hostile headlines must never become Telegram markup.
    hostile = 'See <a href="https://evil.test">this</a> & <b>act now</b>'
    text = format_anomaly(_ANOMALY, _explanation(reasoning=hostile))

    assert "<a href" not in text
    assert "&lt;a href=&quot;https://evil.test&quot;&gt;" in text
    assert "&amp;" in text
    assert "<b>BTC-USD</b>" in text  # our own markup stays literal markup


def test_status_is_honest_when_nothing_explained_yet() -> None:
    assert "No explained anomalies yet" in format_status(None)
    assert "Latest anomaly:" in format_status((_ANOMALY, _explanation()))


# The thin client ------------------------------------------------------------


def _client(client: httpx.AsyncClient, rate: RateLimiter) -> TelegramClient:
    return TelegramClient(client, token="TOKEN", base_url=_BASE, rate=rate, retry=_FAST)


async def test_send_posts_html_message(rate: RateLimiter) -> None:
    async with httpx.AsyncClient() as http, respx.mock:
        route = respx.post(_SEND_URL).mock(
            return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})
        )
        await _client(http, rate).send(42, "<b>hi</b>")

    assert route.call_count == 1
    payload = json.loads(route.calls.last.request.content)
    assert payload["chat_id"] == 42
    assert payload["text"] == "<b>hi</b>"
    assert payload["parse_mode"] == "HTML"


async def test_ok_false_is_a_failure_even_on_http_200(rate: RateLimiter) -> None:
    async with httpx.AsyncClient() as http, respx.mock:
        respx.post(_SEND_URL).mock(
            return_value=httpx.Response(200, json={"ok": False, "description": "chat not found"})
        )
        with pytest.raises(SourceUnavailable) as excinfo:
            await _client(http, rate).send(42, "hi")
    assert excinfo.value.source == "telegram"


async def test_5xx_retries_then_raises_unavailable(rate: RateLimiter) -> None:
    async with httpx.AsyncClient() as http, respx.mock:
        route = respx.post(_SEND_URL).mock(return_value=httpx.Response(502))
        with pytest.raises(SourceUnavailable):
            await _client(http, rate).send(42, "hi")
    assert route.call_count == _FAST.attempts


async def test_get_updates_unwraps_result(rate: RateLimiter) -> None:
    updates = [{"update_id": 7, "message": {"text": "hi"}}]
    async with httpx.AsyncClient() as http, respx.mock:
        route = respx.post(f"{_BASE}/botTOKEN/getUpdates").mock(
            return_value=httpx.Response(200, json={"ok": True, "result": updates})
        )
        got = await _client(http, rate).get_updates(offset=None)

    assert got == updates
    payload = json.loads(route.calls.last.request.content)
    assert "offset" not in payload  # first poll: take whatever is pending
    assert payload["timeout"] > 0  # long poll, not a busy loop


def test_client_satisfies_both_protocol_halves(rate: RateLimiter) -> None:
    client = TelegramClient(httpx.AsyncClient(), token="T", base_url=_BASE, rate=rate)
    assert isinstance(client, Messenger)
    assert isinstance(client, UpdateSource)


# The poller: single-ID allowlist --------------------------------------------

_ALLOWED = 42
_STRANGER = 666


def _update(update_id: int, sender: int, chat: int | None = None) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "from": {"id": sender},
            "chat": {"id": chat if chat is not None else sender},
            "text": "what's up?",
        },
    }


class ScriptedAPI:
    """Serves one scripted batch of updates, then long-polls forever (until
    the test cancels the poller) — deterministic, no timing guesses."""

    def __init__(self, updates: list[dict[str, Any]]) -> None:
        self.batches = [updates]
        self.sent: list[tuple[int, str]] = []
        self.offsets: list[int | None] = []
        self.drained = asyncio.Event()

    async def get_updates(self, offset: int | None) -> list[dict[str, Any]]:
        self.offsets.append(offset)
        if self.batches:
            return self.batches.pop(0)
        self.drained.set()
        await asyncio.Event().wait()  # park forever; cancellation ends the test
        raise AssertionError("unreachable")

    async def send(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))


async def _run_until_drained(api: ScriptedAPI) -> None:
    async def reply_text() -> str:
        return "latest anomaly summary"

    poller = UpdatePoller(api, allowed_chat_id=_ALLOWED, reply_text=reply_text)
    task = asyncio.create_task(poller.run())
    try:
        await asyncio.wait_for(api.drained.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_allowlisted_user_gets_the_reply_and_strangers_get_silence() -> None:
    api = ScriptedAPI([_update(1, _STRANGER), _update(2, _ALLOWED)])

    await _run_until_drained(api)

    assert api.sent == [(_ALLOWED, "latest anomaly summary")]  # one reply, to us only
    assert api.offsets == [None, 3]  # both updates acknowledged, stranger included


async def test_stranger_in_allowed_chat_is_still_rejected() -> None:
    # Sender and chat must *both* match: a group scenario where the chat id
    # looks right but the sender doesn't stays rejected.
    api = ScriptedAPI([_update(1, sender=_STRANGER, chat=_ALLOWED)])

    await _run_until_drained(api)

    assert api.sent == []
