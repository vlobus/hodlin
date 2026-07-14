"""Thin Telegram Bot API client behind Protocol seams (T9).

Two endpoints are all delivery needs — ``sendMessage`` out, ``getUpdates``
in — so this is a hand-rolled client over the shared connector resilience
(rate limit + retry + ``SourceUnavailable``), not a bot framework: Telegram
is a sink that fails exactly like a source, and the jobs already know how to
treat a dead source. ``Messenger`` is the outbound seam the notify job mocks;
``UpdateSource`` is the inbound seam the poller mocks; ``TelegramClient`` is
the one concrete implementing both.

Delivery semantics are at-least-once by construction: the shared retry can
repeat a ``sendMessage`` whose reply was lost after the server acted, and the
notify job commits its claim only after a successful send. A rare duplicate
alert beats a silently missing one.
"""

from typing import Any, Protocol, runtime_checkable

import httpx

from hodlin_recommend.connectors.base import (
    DEFAULT_RETRY,
    RateLimiter,
    RetryPolicy,
    SourceUnavailable,
    request_json,
)

# Tuning, not secrets (D17): how long Telegram holds a getUpdates open. The
# HTTP read timeout must outlive it or every idle poll "times out".
LONG_POLL_S = 50


@runtime_checkable
class Messenger(Protocol):
    """Outbound half — everything the notify job needs."""

    async def send(self, chat_id: int, text: str) -> None: ...


@runtime_checkable
class UpdateSource(Protocol):
    """Inbound half — everything the poller reads. Returns raw Bot API update
    dicts; interpreting them (and enforcing the allowlist) is the poller's job."""

    async def get_updates(self, offset: int | None) -> list[dict[str, Any]]: ...


class TelegramClient:
    """The one concrete: both halves over the HTTP Bot API."""

    source = "telegram"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        token: str,
        base_url: str,
        rate: RateLimiter,
        retry: RetryPolicy = DEFAULT_RETRY,
    ) -> None:
        self._client = client
        # Telegram puts the secret in the path; it must never be logged.
        self._base = f"{base_url.rstrip('/')}/bot{token}"
        self._rate = rate
        self._retry = retry

    async def _call(
        self,
        api_method: str,
        payload: dict[str, Any],
        *,
        http_timeout: httpx.Timeout | None = None,
    ) -> Any:
        data = await request_json(
            self._client,
            source=self.source,
            url=f"{self._base}/{api_method}",
            method="POST",
            json_body=payload,
            http_timeout=http_timeout,
            rate=self._rate,
            retry=self._retry,
        )
        # Telegram can answer HTTP 200 with ok=false; that's still a failure.
        if not isinstance(data, dict) or not data.get("ok"):
            raise SourceUnavailable(self.source, f"API answered not-ok: {data!r}")
        return data.get("result")

    async def send(self, chat_id: int, text: str) -> None:
        await self._call(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
                # HTML parse mode pairs with formatting.py escaping every
                # dynamic field — untrusted text can't become markup.
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
        )

    async def get_updates(self, offset: int | None) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"timeout": LONG_POLL_S, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        result = await self._call(
            "getUpdates", payload, http_timeout=httpx.Timeout(LONG_POLL_S + 10)
        )
        return list(result) if isinstance(result, list) else []
