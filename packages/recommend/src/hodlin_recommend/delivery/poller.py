"""Inbound long-poll loop (T9): thin by design, default-deny by design.

The poller reads raw updates and enforces the single-ID allowlist where
messages *enter* the system: an update whose chat or sender isn't the one
allowlisted ID is dropped without a reply (strangers learn nothing, not even
that the bot exists). An allowlisted message gets the latest explained
anomaly back — the reply text comes through an injected async callable so the
poller itself never touches the database and unit tests never need one.

Runs as a background task owned by the app lifespan; cancellation is the stop
signal. A dead Telegram API is absorbed with a backoff, same policy as any
dead source: the loop must outlive the outage.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from hodlin_recommend.connectors.base import SourceUnavailable
from hodlin_recommend.delivery.formatting import format_status
from hodlin_recommend.delivery.telegram import Messenger, UpdateSource
from hodlin_recommend.store.db import SessionFactory
from hodlin_recommend.store.repositories import AnomalyRepository

# Tuning, not secrets (D17): how long to sit out a Telegram outage before the
# next poll attempt.
ERROR_BACKOFF_S = 5.0


class TelegramAPI(Messenger, UpdateSource, Protocol):
    """Both halves — what the poller needs (read updates, send replies)."""


class UpdatePoller:
    def __init__(
        self,
        api: TelegramAPI,
        *,
        allowed_chat_id: int,
        reply_text: Callable[[], Awaitable[str]],
    ) -> None:
        self._api = api
        self._allowed = allowed_chat_id
        self._reply_text = reply_text

    async def run(self) -> None:
        """Poll forever; the owner cancels this task to stop it. ``offset``
        acknowledges processed updates so Telegram never redelivers them."""
        offset: int | None = None
        while True:
            try:
                updates = await self._api.get_updates(offset)
            except SourceUnavailable:
                await asyncio.sleep(ERROR_BACKOFF_S)
                continue
            for update in updates:
                offset = int(update["update_id"]) + 1
                await self._handle(update)

    async def _handle(self, update: dict[str, Any]) -> None:
        message = update.get("message") or {}
        chat_id = (message.get("chat") or {}).get("id")
        sender_id = (message.get("from") or {}).get("id")
        if chat_id != self._allowed or sender_id != self._allowed:
            return  # default-deny: no reply, no error, no acknowledgement
        try:
            await self._api.send(self._allowed, await self._reply_text())
        except SourceUnavailable:
            return  # the reply is best-effort; the user can just ask again


def latest_anomaly_reply(session_factory: SessionFactory) -> Callable[[], Awaitable[str]]:
    """The production reply builder: a fresh session per inbound message,
    the newest explained anomaly formatted, or an honest 'nothing yet'."""

    async def reply() -> str:
        async with session_factory() as session:
            latest = await AnomalyRepository(session).latest_explained()
        return format_status(latest)

    return reply
