"""initial M1 schema

Revision ID: 0001
Revises:
Create Date: 2026-06-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NOW = sa.text("now()")


def upgrade() -> None:
    op.create_table(
        "assets",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.UniqueConstraint("symbol", name="uq_assets_symbol"),
    )

    op.create_table(
        "price_bars",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("asset_id", sa.BigInteger(), nullable=False),
        sa.Column("interval", sa.String(length=8), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(), nullable=False),
        sa.Column("high", sa.Numeric(), nullable=False),
        sa.Column("low", sa.Numeric(), nullable=False),
        sa.Column("close", sa.Numeric(), nullable=False),
        sa.Column("volume", sa.Numeric(), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("asset_id", "interval", "ts", name="uq_price_bars_natural"),
    )

    op.create_table(
        "news_items",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("asset_id", sa.BigInteger(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("external_id", sa.String(length=128), nullable=False),
        sa.Column("headline", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("source", "external_id", name="uq_news_items_natural"),
    )
    op.create_index("ix_news_items_asset", "news_items", ["asset_id", "published_at"])

    op.create_table(
        "sentiments",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("news_item_id", sa.BigInteger(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=16), nullable=False),
        sa.Column("prob_positive", sa.Numeric(), nullable=False),
        sa.Column("prob_negative", sa.Numeric(), nullable=False),
        sa.Column("prob_neutral", sa.Numeric(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["news_item_id"], ["news_items.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("news_item_id", "model_version", name="uq_sentiments_natural"),
    )

    op.create_table(
        "anomalies",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("asset_id", sa.BigInteger(), nullable=False),
        sa.Column("interval", sa.String(length=8), nullable=False),
        sa.Column("bar_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("z_score", sa.Numeric(), nullable=False),
        sa.Column("return_pct", sa.Numeric(), nullable=False),
        sa.Column("direction", sa.String(length=4), nullable=False),
        sa.Column("window", sa.Integer(), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.Column("notified_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("asset_id", "interval", "bar_ts", name="uq_anomalies_natural"),
    )

    op.create_table(
        "explanations",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("anomaly_id", sa.BigInteger(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=False),
        sa.Column("evidence", postgresql.JSONB(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["anomaly_id"], ["anomalies.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("anomaly_id", name="uq_explanations_anomaly"),
    )

    op.create_table(
        "source_health",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("last_ok_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.UniqueConstraint("source", name="uq_source_health_source"),
    )

    op.create_table(
        "ingest_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("job", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("items", sa.Integer(), server_default="0", nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("ingest_runs")
    op.drop_table("source_health")
    op.drop_table("explanations")
    op.drop_table("anomalies")
    op.drop_table("sentiments")
    op.drop_index("ix_news_items_asset", table_name="news_items")
    op.drop_table("news_items")
    op.drop_table("price_bars")
    op.drop_table("assets")
