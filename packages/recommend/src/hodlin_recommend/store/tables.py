"""ORM table definitions — the single source of truth for the schema.

These classes describe the eight M1 tables; Alembic reads ``Base.metadata`` to
generate/verify the migration that actually runs ``CREATE TABLE`` in Postgres.
Conventions (D2, 1.3): money/prices are ``Numeric`` (exact, never float); every
timestamp is ``timestamptz`` (UTC); idempotent ingestion is enforced by
natural-key unique constraints that the repositories upsert against.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hodlin_recommend.store.db import Base


class Asset(Base):
    """A tracked instrument (stock or crypto). ``symbol`` is the natural key
    everything else references; ``id`` is the surrogate FK target."""

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), unique=True)
    kind: Mapped[str] = mapped_column(String(16))  # "stock" | "crypto"
    name: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PriceBar(Base):
    """One OHLCV candle for an asset at a given interval and instant. The
    natural key ``(asset_id, interval, ts)`` makes re-ingesting the same bar a
    no-op; the hot index serves the "last N bars" read the z-score needs."""

    __tablename__ = "price_bars"
    __table_args__ = (
        UniqueConstraint("asset_id", "interval", "ts", name="uq_price_bars_natural"),
        Index("ix_price_bars_recent", "asset_id", "interval", "ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"))
    interval: Mapped[str] = mapped_column(String(8))  # e.g. "1d", "1h"
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    open: Mapped[Decimal] = mapped_column(Numeric)
    high: Mapped[Decimal] = mapped_column(Numeric)
    low: Mapped[Decimal] = mapped_column(Numeric)
    close: Mapped[Decimal] = mapped_column(Numeric)
    volume: Mapped[Decimal | None] = mapped_column(Numeric)
    source: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class NewsItem(Base):
    """A news article tied to an asset. ``(source, external_id)`` is the
    provider's own identity, so the same article fetched twice stays one row."""

    __tablename__ = "news_items"
    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_news_items_natural"),
        Index("ix_news_items_asset", "asset_id", "published_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"))
    source: Mapped[str] = mapped_column(String(32))
    external_id: Mapped[str] = mapped_column(String(128))
    headline: Mapped[str] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Sentiment(Base):
    """A model's sentiment scoring of one news item. One score per
    ``(news_item, model_version)`` so re-running a model version is idempotent."""

    __tablename__ = "sentiments"
    __table_args__ = (
        UniqueConstraint("news_item_id", "model_version", name="uq_sentiments_natural"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    news_item_id: Mapped[int] = mapped_column(ForeignKey("news_items.id", ondelete="CASCADE"))
    model_version: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(16))  # positive | negative | neutral
    prob_positive: Mapped[Decimal] = mapped_column(Numeric)
    prob_negative: Mapped[Decimal] = mapped_column(Numeric)
    prob_neutral: Mapped[Decimal] = mapped_column(Numeric)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Anomaly(Base):
    """A bar that tripped the z-score detector. ``(asset_id, interval, bar_ts)``
    is unique so a bar trips at most once; ``notified_at`` lets Telegram send
    each anomaly exactly once."""

    __tablename__ = "anomalies"
    __table_args__ = (
        UniqueConstraint("asset_id", "interval", "bar_ts", name="uq_anomalies_natural"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"))
    interval: Mapped[str] = mapped_column(String(8))
    bar_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    z_score: Mapped[Decimal] = mapped_column(Numeric)
    return_pct: Mapped[Decimal] = mapped_column(Numeric)
    direction: Mapped[str] = mapped_column(String(4))  # "up" | "down"
    window: Mapped[int] = mapped_column(Integer)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    explanation: Mapped[Explanation | None] = relationship(back_populates="anomaly")


class Explanation(Base):
    """The LLM-authored "why" for an anomaly. One per anomaly (unique FK);
    ``evidence`` is the structured list of EvidenceRef the proposal cites."""

    __tablename__ = "explanations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    anomaly_id: Mapped[int] = mapped_column(
        ForeignKey("anomalies.id", ondelete="CASCADE"), unique=True
    )
    reasoning: Mapped[str] = mapped_column(Text)
    evidence: Mapped[list[dict[str, object]]] = mapped_column(JSONB)
    model_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    anomaly: Mapped[Anomaly] = relationship(back_populates="explanation")


class SourceHealth(Base):
    """Liveness of each connector. One row per ``source``; updated as ingestion
    succeeds or degrades, so the bot can serve last-known data and flag a source."""

    __tablename__ = "source_health"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source: Mapped[str] = mapped_column(String(32), unique=True)
    status: Mapped[str] = mapped_column(String(16))  # ok | degraded | down
    last_ok_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class IngestRun(Base):
    """An audit row per scheduled job execution — proves jobs fire and lets us
    see overlap/failures. Append-only; no natural-key dedupe (each run is real)."""

    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    job: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))  # running | ok | error
    items: Mapped[int] = mapped_column(Integer, server_default="0")
    detail: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
