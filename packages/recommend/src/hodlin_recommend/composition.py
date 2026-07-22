"""Composition root helper (T10): build every concrete once from ``Settings``.

Extracted from ``main.py`` so the serving app and the one-shot demo wire the
same graph — the only place concrete classes are named. Two providers vary
with ``demo_mode``: the offline seed bars (which carry the demo anomaly) and a
null news source replace the live Massive/Finnhub connectors, so a clean
machine runs the full pipeline without those keys. Everything else — FinBERT,
the Anthropic explainer, Telegram, the scheduler — is identical to production.

Heavy, once-only objects (FinBERT ~440 MB) are built here; callers own the
returned ``resources`` exit stack and must ``aclose`` it at shutdown.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import dataclass

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from hodlin_recommend.config import Settings
from hodlin_recommend.connectors.base import NewsSource, PriceBarSource, RateLimiter
from hodlin_recommend.connectors.finnhub import FinnhubNewsSource
from hodlin_recommend.connectors.massive import MassivePriceBarSource
from hodlin_recommend.connectors.offline import NullNewsSource
from hodlin_recommend.connectors.seed_bars import SeedBarSource
from hodlin_recommend.delivery.poller import UpdatePoller, latest_anomaly_reply
from hodlin_recommend.delivery.telegram import TelegramClient
from hodlin_recommend.domain.explanation import AnthropicExplainer, ExplainerLLM
from hodlin_recommend.domain.sentiment import FinBertModel, SentimentModel
from hodlin_recommend.ingest.scheduler import build_scheduler
from hodlin_recommend.store.db import SessionFactory, create_engine, create_session_factory


def select_bar_source(settings: Settings, client: httpx.AsyncClient) -> PriceBarSource:
    """Offline seed bars in demo mode, else the live Massive provider."""
    if settings.demo_mode:
        return SeedBarSource()
    return MassivePriceBarSource(
        client,
        api_key=settings.massive_api_key,
        base_url=settings.massive_base_url,
        rate=RateLimiter(settings.massive_rate_per_min),
    )


def select_news_source(settings: Settings, client: httpx.AsyncClient) -> NewsSource:
    """No news in demo mode (the anomaly self-cites), else live Finnhub."""
    if settings.demo_mode:
        return NullNewsSource()
    return FinnhubNewsSource(
        client,
        api_key=settings.finnhub_api_key,
        base_url=settings.finnhub_base_url,
        rate=RateLimiter(settings.finnhub_rate_per_min),
    )


@dataclass
class Components:
    """The wired graph. ``resources`` holds async cleanup (engine, HTTP client)
    the caller must close last, after the scheduler has stopped."""

    settings: Settings
    session_factory: SessionFactory
    bar_source: PriceBarSource
    news_source: NewsSource
    sentiment_model: SentimentModel
    llm: ExplainerLLM
    messenger: TelegramClient
    chat_id: int
    poller: UpdatePoller
    scheduler: AsyncIOScheduler
    inference_executor: ThreadPoolExecutor
    resources: AsyncExitStack


def build_components(settings: Settings) -> Components:
    """Construct and wire everything from settings — the composition root."""
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)

    client = httpx.AsyncClient()
    bar_source = select_bar_source(settings, client)
    news_source = select_news_source(settings, client)

    telegram = TelegramClient(
        client,
        token=settings.telegram_bot_token,
        base_url=settings.telegram_base_url,
        rate=RateLimiter(settings.telegram_rate_per_min),
    )
    poller = UpdatePoller(
        telegram,
        allowed_chat_id=settings.telegram_chat_id,
        reply_text=latest_anomaly_reply(session_factory),
    )

    sentiment_model = FinBertModel()
    llm = AnthropicExplainer(api_key=settings.anthropic_api_key, model=settings.anthropic_model)
    inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

    scheduler = build_scheduler(
        session_factory=session_factory,
        bar_source=bar_source,
        news_source=news_source,
        llm=llm,
        sentiment_model=sentiment_model,
        messenger=telegram,
        chat_id=settings.telegram_chat_id,
        inference_executor=inference_executor,
    )

    resources = AsyncExitStack()
    resources.push_async_callback(engine.dispose)
    resources.push_async_callback(client.aclose)

    return Components(
        settings=settings,
        session_factory=session_factory,
        bar_source=bar_source,
        news_source=news_source,
        sentiment_model=sentiment_model,
        llm=llm,
        messenger=telegram,
        chat_id=settings.telegram_chat_id,
        poller=poller,
        scheduler=scheduler,
        inference_executor=inference_executor,
        resources=resources,
    )
