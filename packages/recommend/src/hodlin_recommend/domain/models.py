"""Pure domain models passed across the store boundary.

These are framework-light value objects (frozen Pydantic). They are keyed by
``symbol``, never by database id — the domain doesn't know surrogate keys; the
repositories resolve ``symbol -> asset_id`` on the way in and out. Money is
``Decimal`` and timestamps are tz-aware, matching the store's invariants.
"""

from decimal import Decimal
from typing import Annotated

from hodlin_contracts import EvidenceRef
from pydantic import AwareDatetime, BaseModel, BeforeValidator, ConfigDict, Field


def _reject_float(value: object) -> object:
    """Prices must never originate from a float — that reintroduces binary
    rounding error. A string or Decimal parses exactly (mirrors the contracts
    package's money rule)."""
    if isinstance(value, float):
        raise ValueError("price must be a Decimal or string, not a float")
    return value


# Exact money/price; tz-aware timestamps only (DB columns are timestamptz).
Money = Annotated[Decimal, BeforeValidator(_reject_float)]


class _DomainModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Asset(_DomainModel):
    symbol: str
    kind: str  # "stock" | "crypto"
    name: str | None = None


class PriceBar(_DomainModel):
    symbol: str
    interval: str
    ts: AwareDatetime
    open: Money
    high: Money
    low: Money
    close: Money
    volume: Money | None = None
    source: str


class NewsItem(_DomainModel):
    symbol: str
    source: str
    external_id: str
    headline: str
    published_at: AwareDatetime
    url: str | None = None
    summary: str | None = None


class Anomaly(_DomainModel):
    """A bar the z-score detector flagged (D6). ``z_score``/``return_pct`` are
    statistics, not money, but stay ``Decimal`` so they round-trip the store's
    ``Numeric`` columns without float drift."""

    symbol: str
    interval: str
    bar_ts: AwareDatetime
    z_score: Decimal
    return_pct: Decimal
    direction: str  # "up" | "down"
    window: int


class Explanation(_DomainModel):
    """The LLM-authored "why" for one anomaly, keyed by the anomaly's natural
    key (the repository resolves the surrogate FK). ``evidence`` uses the
    shared contract type so it can be lifted into a ``Proposal`` unchanged;
    >= 1 entry is guaranteed (the anomaly cites itself)."""

    symbol: str
    interval: str
    bar_ts: AwareDatetime
    reasoning: str = Field(min_length=1)
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    model_version: str

    model_config = ConfigDict(frozen=True, protected_namespaces=())
