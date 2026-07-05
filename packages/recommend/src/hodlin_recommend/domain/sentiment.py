"""Financial sentiment scoring behind a swappable model seam.

``SentimentModel`` is the Protocol the serving layer depends on — the real
FinBERT adapter, a future fine-tuned replacement, or a test fake all satisfy
it. ``score`` is synchronous and CPU-bound by design: the caller decides how
to keep it off the event loop (serving uses ``asyncio.to_thread``), so the
model stays runnable from sync contexts too.

``FinBertModel`` is the one heavyweight class: constructing it downloads/loads
ProsusAI/finbert (~440 MB, cached by Hugging Face after the first run), so it
is built exactly once at the composition root and shared across requests.
"""

from collections.abc import Mapping
from decimal import Decimal
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

FINBERT_MODEL_ID = "ProsusAI/finbert"

_LABELS = ("positive", "negative", "neutral")

# Probabilities are statistics, not money, but they land in NUMERIC columns and
# JSON payloads — quantized Decimal keeps them exact and readable end to end.
_PROB_PLACES = Decimal("0.000001")


class SentimentScore(BaseModel):
    """One model's verdict on one text. ``model_version`` makes every score
    traceable to the exact model that produced it (lineage, D2)."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    label: str  # "positive" | "negative" | "neutral"
    prob_positive: Decimal
    prob_negative: Decimal
    prob_neutral: Decimal
    model_version: str


def to_score(probs: Mapping[str, float], model_version: str) -> SentimentScore:
    """Pure mapping from raw class probabilities to a ``SentimentScore``:
    label = argmax, floats quantized once at this boundary. Raises if the
    model didn't produce exactly the three expected classes."""
    if set(probs) != set(_LABELS):
        raise ValueError(f"expected probabilities for {_LABELS}, got {sorted(probs)}")
    label = max(_LABELS, key=lambda name: probs[name])
    as_decimal = {name: Decimal(f"{probs[name]:f}").quantize(_PROB_PLACES) for name in _LABELS}
    return SentimentScore(
        label=label,
        prob_positive=as_decimal["positive"],
        prob_negative=as_decimal["negative"],
        prob_neutral=as_decimal["neutral"],
        model_version=model_version,
    )


@runtime_checkable
class SentimentModel(Protocol):
    """The seam serving depends on; concrete models are injected, never imported."""

    model_version: str

    def score(self, text: str) -> SentimentScore: ...


class FinBertModel:
    """FinBERT (ProsusAI/finbert) — pretrained financial-news sentiment.

    Heavy imports happen inside ``__init__`` so merely importing this module
    (or anything in the domain package) never pulls torch/transformers; only
    actually constructing the model pays that cost, once.
    """

    def __init__(self, model_id: str = FINBERT_MODEL_ID) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._model.eval()
        self.model_version = f"finbert:{model_id}"

    def score(self, text: str) -> SentimentScore:
        # Tokenizer truncation caps input at BERT's 512-token window; longer
        # news bodies are scored on their head, which carries the signal.
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probabilities = self._torch.softmax(logits, dim=-1).tolist()
        id2label = self._model.config.id2label
        by_label = {str(id2label[i]).lower(): p for i, p in enumerate(probabilities)}
        return to_score(by_label, self.model_version)
