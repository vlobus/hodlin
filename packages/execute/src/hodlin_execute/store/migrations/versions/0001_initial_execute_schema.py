"""initial execute-domain schema (T11)

RBAC (operators/roles/permissions), the approval trail, the single-use token
replay store, and on-chain attempt records. Alembic owns the schema; this is
hand-written to match ``store/tables.py`` and is proven equivalent + reversible
by tests/integration/test_execute_migrations.py.

Revision ID: 0001
Revises:
Create Date: 2026-08-17
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
# uint256 tops out near 1.16e77 — 78 digits, scale 0, so wei stays exact.
_WEI = sa.Numeric(78, 0)


def upgrade() -> None:
    op.create_table(
        "operators",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.UniqueConstraint("subject", name="uq_operators_subject"),
    )

    op.create_table(
        "roles",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.UniqueConstraint("name", name="uq_roles_name"),
    )

    op.create_table(
        "permissions",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.UniqueConstraint("name", name="uq_permissions_name"),
    )

    op.create_table(
        "role_permissions",
        sa.Column("role_id", sa.BigInteger(), nullable=False),
        sa.Column("permission_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["permission_id"], ["permissions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("role_id", "permission_id"),
    )

    op.create_table(
        "operator_roles",
        sa.Column("operator_id", sa.BigInteger(), nullable=False),
        sa.Column("role_id", sa.BigInteger(), nullable=False),
        sa.Column("granted_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["operator_id"], ["operators.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("operator_id", "role_id"),
    )

    op.create_table(
        "proposals",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("proposal_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("schema_version", sa.String(length=8), nullable=False),
        sa.Column("canonical_hash", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.UniqueConstraint("proposal_id", name="uq_proposals_proposal_id"),
    )
    op.create_index("ix_proposals_canonical_hash", "proposals", ["canonical_hash"])

    op.create_table(
        "auth_tokens",
        sa.Column("jti", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("proposal_hash", sa.String(length=64), nullable=False),
        sa.Column("operator_id", sa.BigInteger(), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consumed_reason", sa.String(length=16), nullable=True),
        sa.ForeignKeyConstraint(["operator_id"], ["operators.id"], ondelete="RESTRICT"),
    )
    # At most one *live* token per proposal, enforced by the database rather
    # than by a code path someone could forget: re-approval must supersede the
    # previous token first.
    op.create_index(
        "uq_auth_tokens_one_live_per_proposal",
        "auth_tokens",
        ["proposal_hash"],
        unique=True,
        postgresql_where=sa.text("consumed_at IS NULL"),
    )
    op.create_index("ix_auth_tokens_operator", "auth_tokens", ["operator_id"])

    op.create_table(
        "approvals",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        # The surrogate proposals.id — named apart from proposals.proposal_id
        # (the contract's UUID) so the two can't be swapped by a caller.
        sa.Column("proposal_row_id", sa.BigInteger(), nullable=False),
        sa.Column("operator_id", sa.BigInteger(), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("token_jti", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["proposal_row_id"], ["proposals.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["operator_id"], ["operators.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["token_jti"], ["auth_tokens.jti"], ondelete="SET NULL"),
    )
    op.create_index("ix_approvals_proposal", "approvals", ["proposal_row_id", "decided_at"])

    op.create_table(
        "tx_attempts",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("proposal_row_id", sa.BigInteger(), nullable=False),
        sa.Column("token_jti", postgresql.UUID(as_uuid=True), nullable=False),
        # 64-bit: EIP-155 chain ids aren't bounded by 2**31, and this row is
        # written before the broadcast, where an overflow is unaffordable.
        sa.Column("chain_id", sa.BigInteger(), nullable=False),
        sa.Column("to_address", sa.String(length=42), nullable=False),
        sa.Column("recipient_label", sa.String(length=64), nullable=False),
        sa.Column("value_wei", _WEI, nullable=False),
        sa.Column("gas_limit", sa.BigInteger(), nullable=True),
        sa.Column("max_fee_per_gas_wei", _WEI, nullable=True),
        sa.Column("max_priority_fee_per_gas_wei", _WEI, nullable=True),
        sa.Column("nonce", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("tx_hash", sa.String(length=66), nullable=True),
        sa.Column("block_number", sa.BigInteger(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.Column("broadcast_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["proposal_row_id"], ["proposals.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["token_jti"], ["auth_tokens.jti"], ondelete="RESTRICT"),
        # Structural single-use insurance: one attempt per token, even if the
        # replay store were bypassed.
        sa.UniqueConstraint("token_jti", name="uq_tx_attempts_token_jti"),
        sa.UniqueConstraint("tx_hash", name="uq_tx_attempts_tx_hash"),
    )
    op.create_index("ix_tx_attempts_status", "tx_attempts", ["status", "created_at"])


def downgrade() -> None:
    op.drop_table("tx_attempts")
    op.drop_table("approvals")
    op.drop_table("auth_tokens")
    op.drop_table("proposals")
    op.drop_table("operator_roles")
    op.drop_table("role_permissions")
    op.drop_table("permissions")
    op.drop_table("roles")
    op.drop_table("operators")
