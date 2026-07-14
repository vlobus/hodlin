"""Explain one anomaly: gather context, one LLM call, store the result.

Orchestration only — the judgement lives in ``domain/explanation.py``. Flow:
recent news for the asset (window ending at the anomalous bar's close, so
later news can't "explain" an earlier move) -> sentiment score
per headline -> numbered candidates -> single LLM call -> validate -> expand
index selections into EvidenceRefs -> persist. Idempotent: an anomaly that
already has an explanation is skipped before any model or LLM work is spent.

Sentiment scoring is sequential and off the event loop. When the scheduler
(T8) wires this job into the same process as serving, pass the app's
single-lane inference executor: the job and HTTP requests then share one
FinBERT instance through one thread, which is what makes concurrent use safe
(CPU oversubscription + tokenizer thread-safety, D18). Standalone callers
(tests) can omit it and get the loop's default pool.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from hodlin_recommend.domain.explanation import (
    ExplainerLLM,
    ScoredNews,
    assemble_explanation,
    build_prompt,
    parse_reply,
)
from hodlin_recommend.domain.models import Anomaly, Explanation
from hodlin_recommend.domain.sentiment import SentimentModel
from hodlin_recommend.store.repositories import ExplanationRepository, NewsRepository

# Tuning, not secrets (D17): how much context one explanation considers.
NEWS_WINDOW = timedelta(days=3)
BAR_SPAN = timedelta(days=1)  # "1d" bars — news during the anomalous bar still counts
MAX_NEWS = 8


async def explain_anomaly(
    session: AsyncSession,
    anomaly: Anomaly,
    *,
    llm: ExplainerLLM,
    sentiment_model: SentimentModel,
    inference_executor: ThreadPoolExecutor | None = None,
) -> Explanation | None:
    """Produce and store the explanation for ``anomaly``; returns it, or
    ``None`` if the anomaly is already explained (nothing was spent or
    stored). Raises ``LLMUnavailable`` / ``MalformedReply`` for the caller
    (the T8 job) to catch and retry on a later tick. Commits on success —
    an explanation is one unit of work."""
    explanations = ExplanationRepository(session)
    if await explanations.for_anomaly(anomaly.symbol, anomaly.interval, anomaly.bar_ts):
        return None

    news = await NewsRepository(session).recent_for_symbol(
        anomaly.symbol,
        since=anomaly.bar_ts - NEWS_WINDOW,
        until=anomaly.bar_ts + BAR_SPAN,
        limit=MAX_NEWS,
    )
    loop = asyncio.get_running_loop()
    candidates = [
        ScoredNews(
            item=item,
            score=await loop.run_in_executor(
                inference_executor, sentiment_model.score, item.headline
            ),
        )
        for item in news
    ]

    system, user = build_prompt(anomaly, candidates)
    reply = await llm.complete(system=system, user=user)
    draft = parse_reply(reply, candidate_count=len(candidates))
    reasoning, evidence = assemble_explanation(anomaly, candidates, draft)

    explanation = Explanation(
        symbol=anomaly.symbol,
        interval=anomaly.interval,
        bar_ts=anomaly.bar_ts,
        reasoning=reasoning,
        evidence=tuple(evidence),
        model_version=llm.model_version,
    )
    await explanations.upsert(explanation)
    await session.commit()
    return explanation
