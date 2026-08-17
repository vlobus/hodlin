"""Offline stand-ins for the demo — no network, no keys.

The seed *bars* live in ``seed_bars.py`` (they carry the demo anomaly); news
has no committed fixture, so demo mode uses this no-op source rather than a
live Finnhub call that would just fail without a key. An explanation with zero
news still works — the anomaly cites itself (T7) — so the demo reaches Telegram
without any news at all.
"""

from datetime import datetime

from hodlin_recommend.domain.models import NewsItem


class NullNewsSource:
    """A ``NewsSource`` that always returns nothing. Health is trivially ok —
    there is no dependency to be down."""

    source = "null"

    async def get_news(self, symbol: str, since: datetime) -> list[NewsItem]:
        return []

    async def health(self) -> bool:
        return True
