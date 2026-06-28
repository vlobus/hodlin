"""Pure domain models passed across the store boundary.

These are framework-light value objects (frozen Pydantic). They are keyed by
``symbol``, never by database id — the domain doesn't know surrogate keys; the
repositories resolve ``symbol -> asset_id`` on the way in and out. Money is
``Decimal`` and timestamps are tz-aware, matching the store's invariants.
"""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class _DomainModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Asset(_DomainModel):
    symbol: str
    kind: str  # "stock" | "crypto"
    name: str | None = None


class PriceBar(_DomainModel):
    symbol: str
    interval: str
    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal | None = None
    source: str


class NewsItem(_DomainModel):
    symbol: str
    source: str
    external_id: str
    headline: str
    published_at: datetime
    url: str | None = None
    summary: str | None = None
