"""POST /v1/sentiment — score one text with the injected sentiment model.

Inference is synchronous CPU-bound work, so it runs via ``asyncio.to_thread``:
the event loop stays free to serve other requests (and the scheduler's jobs)
while a worker thread grinds through the forward pass. The model arrives
through DI, never a module-level import — tests inject a fake and the endpoint
logic is exercised without torch ever loading.
"""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from hodlin_recommend.domain.sentiment import SentimentModel
from hodlin_recommend.serving.schemas import SentimentProbs, SentimentRequest, SentimentResponse

router = APIRouter(prefix="/v1")


def get_sentiment_model(request: Request) -> SentimentModel:
    model: SentimentModel = request.app.state.sentiment_model
    return model


@router.post("/sentiment")
async def score_sentiment(
    payload: SentimentRequest,
    model: Annotated[SentimentModel, Depends(get_sentiment_model)],
) -> SentimentResponse:
    score = await asyncio.to_thread(model.score, payload.text)
    return SentimentResponse(
        label=score.label,
        probs=SentimentProbs(
            positive=score.prob_positive,
            negative=score.prob_negative,
            neutral=score.prob_neutral,
        ),
        model_version=score.model_version,
    )
