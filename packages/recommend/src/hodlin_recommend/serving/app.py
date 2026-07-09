"""FastAPI application factory.

The factory takes its dependencies as arguments instead of building them: the
composition root (``main.py``) constructs the expensive, once-only pieces and
passes them in; tests pass fakes or omit what they don't exercise. The app
owns no globals — everything shared lives on ``app.state``.

The lifespan owns start/stop of what it's given: the scheduler starts after
the app is ready to serve and shuts down first (no new ticks while the app
drains), then the inference executor, then ``resources`` — the exit stack the
composition root filled with engine/client cleanup.

Inference runs on a dedicated single-worker executor, not the default
``to_thread`` pool (D18): Hugging Face fast tokenizers aren't thread-safe
while (re)configuring truncation state, and N parallel forward passes
oversubscribe the CPU. The scheduler's explain job shares this same executor,
so *all* inference in the process goes through one lane. Created eagerly (not
in the lifespan) so test clients that skip lifespan startup still work; when
the composition root hands one in, the app takes ownership of its shutdown.
"""

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Protocol, runtime_checkable

from fastapi import FastAPI

from hodlin_recommend.domain.sentiment import SentimentModel
from hodlin_recommend.ingest.jobs import SessionFactory
from hodlin_recommend.serving.routes import health, sentiment


@runtime_checkable
class SchedulerLike(Protocol):
    """What the app needs from a scheduler — start/stop and a liveness flag.
    Satisfied by ``AsyncIOScheduler``; tests satisfy it with a fake."""

    @property
    def running(self) -> bool: ...

    def start(self) -> None: ...

    def shutdown(self, wait: bool = True) -> None: ...


def create_app(
    *,
    sentiment_model: SentimentModel,
    inference_executor: ThreadPoolExecutor | None = None,
    scheduler: SchedulerLike | None = None,
    session_factory: SessionFactory | None = None,
    resources: AsyncExitStack | None = None,
) -> FastAPI:
    executor = inference_executor or ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="inference"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if scheduler is not None:
            scheduler.start()
        try:
            yield
        finally:
            if scheduler is not None:
                # AsyncIOScheduler defers the actual stop to a loop callback;
                # yield once so it has really stopped before the executor and
                # the engine/client (resources) disappear under a live job.
                scheduler.shutdown(wait=True)
                await asyncio.sleep(0)
            executor.shutdown(wait=True)
            if resources is not None:
                await resources.aclose()

    app = FastAPI(title="hodlin recommend", version="0.1.0", lifespan=lifespan)
    app.state.sentiment_model = sentiment_model
    app.state.inference_executor = executor
    app.state.scheduler = scheduler
    app.state.session_factory = session_factory
    app.include_router(health.router)
    app.include_router(sentiment.router)
    return app
