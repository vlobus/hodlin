"""FastAPI application factory.

The factory takes its dependencies as arguments instead of building them: the
composition root (``main.py``) constructs the expensive, once-only pieces and
passes them in; tests pass fakes or omit what they don't exercise. The app
owns no globals — everything shared lives on ``app.state``.

The lifespan owns start/stop of what it's given: the scheduler stops first
(no new ticks while the app drains), then the inference executor, then
``resources`` — the exit stack the composition root filled with engine/client
cleanup. APScheduler 3.x cannot await its own async jobs from inside the
loop, so a tick still in flight at shutdown is *cancelled*, not finished; a
short grace lets it unwind its session before the engine goes away, and a
tick cut off there leaves a visible "running" audit row whose work the next
startup simply redoes (every job is idempotent).

Inference runs on a dedicated single-worker executor, not the default
``to_thread`` pool (D18): Hugging Face fast tokenizers aren't thread-safe
while (re)configuring truncation state, and N parallel forward passes
oversubscribe the CPU. The scheduler's explain job shares this same executor,
so *all* inference in the process goes through one lane. Created eagerly (not
in the lifespan) so test clients that skip lifespan startup still work; when
the composition root hands one in, the app takes ownership of its shutdown.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Protocol, runtime_checkable

from fastapi import FastAPI

from hodlin_recommend.domain.sentiment import SentimentModel
from hodlin_recommend.serving.routes import health, sentiment
from hodlin_recommend.store.db import SessionFactory

# How long a tick cancelled at shutdown gets to unwind before its resources
# close. Tuning, not secrets (D17).
SHUTDOWN_GRACE_S = 0.25


@runtime_checkable
class SchedulerLike(Protocol):
    """What the app needs from a scheduler — start/stop and a liveness flag.
    Satisfied by ``AsyncIOScheduler``; tests satisfy it with a fake."""

    @property
    def running(self) -> bool: ...

    def start(self) -> None: ...

    def shutdown(self, wait: bool = True) -> None: ...


@runtime_checkable
class PollerLike(Protocol):
    """A run-forever loop the lifespan owns as a background task; cancelling
    the task is the stop signal. Satisfied by ``UpdatePoller``."""

    async def run(self) -> None: ...


def create_app(
    *,
    sentiment_model: SentimentModel,
    inference_executor: ThreadPoolExecutor | None = None,
    scheduler: SchedulerLike | None = None,
    poller: PollerLike | None = None,
    session_factory: SessionFactory | None = None,
    resources: AsyncExitStack | None = None,
) -> FastAPI:
    executor = inference_executor or ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="inference"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        poll_task: asyncio.Task[None] | None = None
        try:
            if scheduler is not None:
                scheduler.start()
            if poller is not None:
                poll_task = asyncio.create_task(poller.run(), name="telegram-poller")
            yield
        finally:
            if poll_task is not None:
                # Cancellation is the poller's stop signal; await the unwind
                # so no inbound handler is mid-flight when resources close.
                # Suppress Exception too: a task that somehow crashed earlier
                # re-raises here, and it must not abort the rest of teardown.
                poll_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await poll_task
            if scheduler is not None:
                # AsyncIOScheduler defers the actual stop to a loop callback
                # and cancels (not awaits) in-flight ticks: yield once so the
                # stop and the cancellations land, then a bounded grace so a
                # cancelled tick can unwind its DB session before the
                # engine/client (resources) disappear beneath it.
                scheduler.shutdown(wait=True)
                await asyncio.sleep(0)
                await asyncio.sleep(SHUTDOWN_GRACE_S)
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
