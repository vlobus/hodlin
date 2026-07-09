"""Composition root — the one place concretes are constructed and wired.

Everything expensive or secret is built exactly once, here: settings from the
environment (fail-fast, no defaults), the DB engine, one shared HTTP client,
both connectors, FinBERT (~440 MB on first run, then the HF cache), the
Anthropic explainer, the single-lane inference executor, and the scheduler
that ties the jobs to all of it. The app factory and jobs see only interfaces.

Cleanup that isn't owned by a component goes on the exit stack: the lifespan
closes it last, after the scheduler has stopped taking new ticks and any
cancelled in-flight tick has had its grace to unwind (see ``serving/app.py``
for the honest shutdown semantics). Run with ``python -m hodlin_recommend.main``.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack

import httpx
import uvicorn

from hodlin_recommend.config import Settings
from hodlin_recommend.connectors.base import RateLimiter
from hodlin_recommend.connectors.finnhub import FinnhubNewsSource
from hodlin_recommend.connectors.massive import MassivePriceBarSource
from hodlin_recommend.domain.explanation import AnthropicExplainer
from hodlin_recommend.domain.sentiment import FinBertModel
from hodlin_recommend.ingest.scheduler import build_scheduler
from hodlin_recommend.serving.app import create_app
from hodlin_recommend.store.db import create_engine, create_session_factory


def main() -> None:
    # Fields arrive from the environment at runtime; mypy can't see that.
    settings = Settings()  # type: ignore[call-arg]
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)

    client = httpx.AsyncClient()
    news_source = FinnhubNewsSource(
        client,
        api_key=settings.finnhub_api_key,
        base_url=settings.finnhub_base_url,
        rate=RateLimiter(settings.finnhub_rate_per_min),
    )
    bar_source = MassivePriceBarSource(
        client,
        api_key=settings.massive_api_key,
        base_url=settings.massive_base_url,
        rate=RateLimiter(settings.massive_rate_per_min),
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
        inference_executor=inference_executor,
    )

    resources = AsyncExitStack()
    resources.push_async_callback(engine.dispose)
    resources.push_async_callback(client.aclose)

    app = create_app(
        sentiment_model=sentiment_model,
        inference_executor=inference_executor,
        scheduler=scheduler,
        session_factory=session_factory,
        resources=resources,
    )
    # Loopback bind for local runs; T10's compose/Dockerfile overrides this.
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
