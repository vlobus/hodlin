"""Connector Protocols, rate limiting, retries, and the shared HTTP helper.

The two Protocols are the dependency-inversion seam: jobs and the scheduler
depend on ``NewsSource`` / ``PriceBarSource``, so a concrete provider (Finnhub,
Massive, or a seed CSV) is injected and can be swapped or mocked freely.

``request_json`` centralises the resilience policy every HTTP connector shares:
acquire a rate-limit slot, retry transient failures (network errors, timeouts,
429, 5xx) with exponential backoff + jitter, and on exhaustion — or on a
non-retryable error like 403 — raise ``SourceUnavailable`` so the caller can
fall back to stored data and keep the bot alive.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

import httpx
from aiolimiter import AsyncLimiter
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from hodlin_recommend.domain.models import NewsItem, PriceBar


class SourceUnavailable(Exception):
    """A data source could not be reached (after retries) or refused the request.
    Jobs catch this to degrade gracefully rather than crash."""

    def __init__(self, source: str, cause: BaseException | str) -> None:
        super().__init__(f"source {source!r} unavailable: {cause}")
        self.source = source
        self.cause = cause


@runtime_checkable
class NewsSource(Protocol):
    """A provider of news articles for a symbol since a given time."""

    source: str

    async def get_news(self, symbol: str, since: datetime) -> list[NewsItem]: ...

    async def health(self) -> bool: ...


@runtime_checkable
class PriceBarSource(Protocol):
    """A provider of OHLC price bars for a symbol/interval over a time range."""

    source: str

    async def get_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[PriceBar]: ...

    async def health(self) -> bool: ...


class RateLimiter:
    """Async token-bucket limiter (thin wrapper over ``aiolimiter``). Used as
    ``async with limiter:`` to hold a request under a provider's per-minute cap."""

    def __init__(self, max_rate: float, time_period: float = 60.0) -> None:
        self._limiter = AsyncLimiter(max_rate, time_period)

    async def __aenter__(self) -> RateLimiter:
        await self._limiter.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


@dataclass(frozen=True)
class RetryPolicy:
    """How hard to retry a transient failure before giving up. Tests pass a tiny
    ``wait_initial`` so backoff doesn't slow the suite."""

    attempts: int = 3
    wait_initial: float = 0.2
    wait_max: float = 2.0


# Shared immutable default — safe as a function-argument default (frozen).
DEFAULT_RETRY = RetryPolicy()


def _is_retryable(exc: BaseException) -> bool:
    """Transient failures worth retrying: network/timeout errors, 429, and 5xx.
    A 4xx like 403 (a bad/blocked key) is permanent — fail fast, don't hammer."""
    if isinstance(exc, httpx.TimeoutException | httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


async def request_json(
    client: httpx.AsyncClient,
    *,
    source: str,
    url: str,
    params: Mapping[str, str] | None = None,
    rate: RateLimiter | None = None,
    retry: RetryPolicy = DEFAULT_RETRY,
) -> Any:
    """GET ``url`` and return parsed JSON, applying the shared rate-limit + retry
    policy and wrapping any HTTP failure as ``SourceUnavailable``."""

    async def _once() -> Any:
        if rate is not None:
            async with rate:
                response = await client.get(url, params=params)
        else:
            response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(retry.attempts),
            wait=wait_exponential_jitter(initial=retry.wait_initial, max=retry.wait_max),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                return await _once()
    except httpx.HTTPError as exc:
        raise SourceUnavailable(source, exc) from exc
    # Unreachable: the loop either returns or reraises, but satisfies the type.
    raise SourceUnavailable(source, "retries exhausted")
