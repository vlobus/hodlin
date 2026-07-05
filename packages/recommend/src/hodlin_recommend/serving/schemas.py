"""API DTOs — deliberately separate from domain models.

The wire contract can stay stable (or version) independently of internal
refactors, and request validation happens here, before anything reaches the
domain. Decimals serialize as JSON strings, so probabilities cross the wire
exactly as stored.
"""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class SentimentRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    # min_length after stripping: an all-whitespace text is rejected, not scored.
    text: str = Field(min_length=1, max_length=10_000)


class SentimentProbs(BaseModel):
    positive: Decimal
    negative: Decimal
    neutral: Decimal


class SentimentResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    label: str  # "positive" | "negative" | "neutral"
    probs: SentimentProbs
    model_version: str
