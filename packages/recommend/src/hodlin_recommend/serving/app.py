"""FastAPI application factory.

The factory takes its dependencies as arguments instead of building them: the
composition root constructs the (expensive, once-only) sentiment model and
passes it in; tests pass fakes. The app owns no globals — everything shared
lives on ``app.state``. The lifespan grows the scheduler and Telegram bot in
T8/T9; for now startup is trivially clean.
"""

from fastapi import FastAPI

from hodlin_recommend.domain.sentiment import SentimentModel
from hodlin_recommend.serving.routes import health, sentiment


def create_app(*, sentiment_model: SentimentModel) -> FastAPI:
    app = FastAPI(title="hodlin recommend", version="0.1.0")
    app.state.sentiment_model = sentiment_model
    app.include_router(health.router)
    app.include_router(sentiment.router)
    return app
