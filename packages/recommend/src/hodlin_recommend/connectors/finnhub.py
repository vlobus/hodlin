"""Finnhub news connector (``NewsSource``) — the ``company-news`` endpoint (D4).

Turns Finnhub's article JSON into ``NewsItem`` domain models. All dependencies
(HTTP client, key, base URL, rate limiter) are injected, so tests drive it with
a mocked transport and a fake key — no network, no global config.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import httpx

from hodlin_recommend.connectors.base import (
    DEFAULT_RETRY,
    RateLimiter,
    RetryPolicy,
    SourceUnavailable,
    request_json,
)
from hodlin_recommend.domain.models import NewsItem

_DATE_FMT = "%Y-%m-%d"


def _parse_article(symbol: str, raw: Mapping[str, Any]) -> NewsItem:
    """Map one Finnhub article to a ``NewsItem``. ``datetime`` is unix seconds."""
    return NewsItem(
        symbol=symbol,
        source="finnhub",
        external_id=str(raw["id"]),
        headline=raw["headline"],
        published_at=datetime.fromtimestamp(raw["datetime"], tz=UTC),
        url=raw.get("url") or None,
        summary=raw.get("summary") or None,
    )


class FinnhubNewsSource:
    source = "finnhub"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str,
        base_url: str,
        rate: RateLimiter,
        retry: RetryPolicy = DEFAULT_RETRY,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._rate = rate
        self._retry = retry

    async def get_news(self, symbol: str, since: datetime) -> list[NewsItem]:
        params = {
            "symbol": symbol,
            "from": since.astimezone(UTC).strftime(_DATE_FMT),
            "to": datetime.now(UTC).strftime(_DATE_FMT),
            "token": self._api_key,
        }
        payload = await request_json(
            self._client,
            source=self.source,
            url=f"{self._base_url}/company-news",
            params=params,
            rate=self._rate,
            retry=self._retry,
        )
        if not isinstance(payload, list):
            raise SourceUnavailable(self.source, "expected a JSON array of articles")
        return [_parse_article(symbol, raw) for raw in payload]

    async def health(self) -> bool:
        """Liveness: a today-window news fetch that doesn't raise means healthy."""
        try:
            await self.get_news("AAPL", datetime.now(UTC))
        except SourceUnavailable:
            return False
        return True
