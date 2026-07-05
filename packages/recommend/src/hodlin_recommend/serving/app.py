"""FastAPI application factory.

The factory takes its dependencies as arguments instead of building them: the
composition root (``main.py``) constructs the expensive, once-only sentiment
model and passes it in; tests pass fakes. The app owns no globals — everything
shared lives on ``app.state``. The lifespan grows the scheduler and Telegram
bot in T8/T9.

Inference runs on a dedicated single-worker executor, not the default
``to_thread`` pool: Hugging Face fast tokenizers aren't thread-safe while
(re)configuring truncation state, and N parallel forward passes oversubscribe
the CPU (torch already parallelizes one pass internally). One worker
serializes inference — concurrent requests queue behind it while the event
loop stays free. Created eagerly (not in the lifespan) so test clients that
skip lifespan startup still work; the lifespan only owns the clean shutdown.
"""

from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from hodlin_recommend.domain.sentiment import SentimentModel
from hodlin_recommend.serving.routes import health, sentiment


def create_app(*, sentiment_model: SentimentModel) -> FastAPI:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            executor.shutdown(wait=True)

    app = FastAPI(title="hodlin recommend", version="0.1.0", lifespan=lifespan)
    app.state.sentiment_model = sentiment_model
    app.state.inference_executor = executor
    app.include_router(health.router)
    app.include_router(sentiment.router)
    return app
