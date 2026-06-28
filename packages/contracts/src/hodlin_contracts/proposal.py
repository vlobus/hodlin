"""Frozen, validated contracts exchanged between the two domains (D7, D14).

``Proposal`` is what the recommend domain produces and the execute domain
consumes. It is immutable (``frozen=True``), rejects unknown fields
(``extra="forbid"``), carries money as ``Decimal`` so it stays exact (never a
float), uses timezone-aware datetimes only, and requires at least one piece of
evidence. It deliberately carries **no raw destination address** — the
recipient is named by label and resolved to an address inside the execute
domain at tx-build time, so a prompt-injected recommend domain can't direct
funds anywhere (D14).
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
)

from hodlin_contracts.version import SCHEMA_VERSION


def _reject_float(value: object) -> object:
    """Money must never originate from a float — that reintroduces the
    binary-rounding error the Decimal type exists to avoid. A string or
    Decimal parses exactly; a float does not."""
    if isinstance(value, float):
        raise ValueError("money must be a Decimal or string, not a float")
    return value


def _normalize_money(value: Decimal) -> Decimal:
    """Reduce a money amount to a single canonical form so meaning-equal
    amounts hash identically: ``0.50`` and ``0.5`` must serialize the same.
    ``normalize`` strips trailing zeros but can yield exponent notation for
    whole numbers (``Decimal("1E+2")``); rebuilding from the fixed-point
    string form keeps the canonical value plain (never scientific) without
    ``quantize``, which would raise ``InvalidOperation`` on amounts whose
    digit count exceeds the decimal context precision. ``Decimal`` parses a
    string exactly, free of context limits."""
    normalized = value.normalize()
    if normalized == 0:
        return Decimal(0)  # collapse signed/scaled zeros (e.g. -0, 0E+2) to "0"
    return Decimal(format(normalized, "f"))


def _to_utc(value: datetime) -> datetime:
    """Pin every timestamp to UTC so the same instant has one representation:
    ``12:05Z`` and ``13:05+01:00`` are equal in meaning and must hash alike."""
    return value.astimezone(UTC)


Money = Annotated[Decimal, BeforeValidator(_reject_float), AfterValidator(_normalize_money)]

UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]

Action = Literal["buy", "sell", "hold", "alert"]


class _Frozen(BaseModel):
    """Base config shared by every contract: immutable, no unknown fields,
    surrounding whitespace stripped (so a ``min_length=1`` field can't be
    satisfied by spaces, and canonical strings stay stable)."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class EvidenceRef(_Frozen):
    """A single citable source behind a proposal — a news item, a price
    anomaly, or a sentiment score. At least one is required on every
    proposal so a recommendation can always be traced back to what it saw."""

    kind: Literal["anomaly", "news", "sentiment", "price"]
    source: str = Field(min_length=1)
    ref: str = Field(min_length=1)
    observed_at: UtcDatetime


class Proposal(_Frozen):
    """An AI-authored recommendation. Self-describing and immutable; becomes
    load-bearing (canonical-hashed and token-signed) in slice C."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    proposal_id: UUID
    asset: str = Field(min_length=1)
    action: Action
    amount: Money = Field(ge=0)
    recipient_label: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    created_at: UtcDatetime
