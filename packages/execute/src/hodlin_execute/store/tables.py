"""ORM tables for the execute domain — the schema Alembic owns (T11).

Three groups, matching the three things this domain is trusted to do:

* **who** — ``operators`` / ``roles`` / ``permissions`` and their join tables.
  Identity comes from an OIDC provider we only *verify* (D33); authorization is
  ours, so the grants live here.
* **what was authorized** — ``proposals`` (as received, with the hash *we*
  computed), ``approvals`` (who decided, when), ``auth_tokens`` (the single-use
  replay store: single-use is a compare-and-set on ``consumed_at``, T14).
* **what was attempted on-chain** — ``tx_attempts``, written *before* anything
  leaves the process, so a crash mid-broadcast leaves evidence rather than a
  mystery (the T8/D23 audit-row lesson applied to money).

Conventions carried from M1: every timestamp is ``timestamptz`` (UTC), exact
numerics never float. Wei is an *integer* amount — ``Numeric(78, 0)`` because
uint256 tops out around 1.16e77, and a float would silently misfund a transfer.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from hodlin_execute.store.db import Base

# uint256 needs 78 decimal digits; scale 0 keeps wei a whole number.
Wei = Numeric(78, 0)


class Operator(Base):
    """A human who may act on proposals. ``subject`` is the OIDC ``sub`` claim —
    the only identity assertion we trust, and deliberately not the email, which
    a provider may let a user change."""

    __tablename__ = "operators"
    __table_args__ = (UniqueConstraint("subject", name="uq_operators_subject"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    subject: Mapped[str] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(320))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Role(Base):
    """A named bundle of permissions (e.g. ``approver``)."""

    __tablename__ = "roles"
    __table_args__ = (UniqueConstraint("name", name="uq_roles_name"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    description: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Permission(Base):
    """A single capability, named ``resource:action`` (e.g. ``trade:approve``).
    Code checks permissions, never roles — so re-organising roles never touches
    an authorization check."""

    __tablename__ = "permissions"
    __table_args__ = (UniqueConstraint("name", name="uq_permissions_name"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    description: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RolePermission(Base):
    """Which permissions a role carries. Composite PK — the pair *is* the row."""

    __tablename__ = "role_permissions"

    role_id: Mapped[int] = mapped_column(
        ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True
    )
    permission_id: Mapped[int] = mapped_column(
        ForeignKey("permissions.id", ondelete="CASCADE"), primary_key=True
    )


class OperatorRole(Base):
    """Which roles an operator holds."""

    __tablename__ = "operator_roles"

    operator_id: Mapped[int] = mapped_column(
        ForeignKey("operators.id", ondelete="CASCADE"), primary_key=True
    )
    role_id: Mapped[int] = mapped_column(
        ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True
    )
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProposalRecord(Base):
    """A proposal as this domain received it, plus the canonical hash **this
    domain computed itself**.

    Storing the payload verbatim and re-deriving the hash locally is the point:
    a client-supplied hash would let a caller decide what its own proposal
    "means", and every downstream guarantee (the token binding, the tamper
    check) hangs off that hash. ``payload`` is JSONB so a stored proposal can be
    re-validated against any schema version later (D32).
    """

    __tablename__ = "proposals"
    __table_args__ = (
        UniqueConstraint("proposal_id", name="uq_proposals_proposal_id"),
        Index("ix_proposals_canonical_hash", "canonical_hash"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    proposal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    schema_version: Mapped[str] = mapped_column(String(8))
    canonical_hash: Mapped[str] = mapped_column(String(64))  # hex SHA-256
    payload: Mapped[dict[str, object]] = mapped_column(JSONB)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AuthToken(Base):
    """The replay store: one row per minted authorization token (D15).

    Expiry and single-use are two different mechanisms — ``expires_at`` bounds
    the window, ``consumed_at`` makes it one-shot, and neither substitutes for
    the other. Consumption is an atomic compare-and-set
    (``UPDATE ... WHERE consumed_at IS NULL RETURNING``), which is what makes it
    correct against concurrent *processes* and not merely concurrent requests
    (T14).

    The partial unique index enforces **at most one live token per proposal** in
    the database rather than in a code path that could be forgotten: re-approval
    must first supersede the previous token (stamping ``consumed_at``), so "I
    approved twice and got two spendable tokens" is unrepresentable.
    """

    __tablename__ = "auth_tokens"
    __table_args__ = (
        Index(
            "uq_auth_tokens_one_live_per_proposal",
            "proposal_hash",
            unique=True,
            postgresql_where=text("consumed_at IS NULL"),
        ),
        Index("ix_auth_tokens_operator", "operator_id"),
    )

    jti: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    proposal_hash: Mapped[str] = mapped_column(String(64))
    operator_id: Mapped[int] = mapped_column(ForeignKey("operators.id", ondelete="RESTRICT"))
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Why it stopped being live: "spent" (used for a tx) or "superseded"
    # (replaced by a re-approval). NULL while the token is still live.
    consumed_reason: Mapped[str | None] = mapped_column(String(16))


class Approval(Base):
    """An audited human decision on a proposal.

    Every outcome lands here, including the refusals — a caller who lacked
    ``trade:approve`` is exactly the event worth being able to query later, so
    ``decision`` covers ``approved`` / ``denied`` / ``refused``. ``token_jti`` is
    set only when a token was actually minted.
    """

    __tablename__ = "approvals"
    __table_args__ = (Index("ix_approvals_proposal", "proposal_row_id", "decided_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    # ``proposal_row_id``, not ``proposal_id``: this is the surrogate
    # ``proposals.id``, while ``proposals.proposal_id`` is the contract's UUID.
    # Under one name the two are interchangeable to a reader and to the type
    # checker — ``Approval(proposal_id=proposal.proposal_id, ...)`` would compile
    # and fail at INSERT, in the middle of recording a human's decision.
    proposal_row_id: Mapped[int] = mapped_column(ForeignKey("proposals.id", ondelete="CASCADE"))
    operator_id: Mapped[int] = mapped_column(ForeignKey("operators.id", ondelete="RESTRICT"))
    decision: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text)
    token_jti: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("auth_tokens.jti", ondelete="SET NULL")
    )
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TxAttempt(Base):
    """One attempt to put a transaction on chain, written **before** the
    broadcast (T17).

    Recording the intent first is what makes a crash mid-broadcast diagnosable
    instead of ambiguous: the row says what we were about to send, and the
    reconciler can ask the chain whether it landed. The unique constraint on
    ``token_jti`` is structural single-use insurance — even if the replay store
    were somehow bypassed, a second attempt against the same token cannot be
    inserted.

    ``to_address`` is the address the execute domain **resolved from its own
    allowlist** (D14); the proposal never carries one.
    """

    __tablename__ = "tx_attempts"
    __table_args__ = (
        UniqueConstraint("token_jti", name="uq_tx_attempts_token_jti"),
        UniqueConstraint("tx_hash", name="uq_tx_attempts_tx_hash"),
        Index("ix_tx_attempts_status", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    proposal_row_id: Mapped[int] = mapped_column(ForeignKey("proposals.id", ondelete="RESTRICT"))
    token_jti: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("auth_tokens.jti", ondelete="RESTRICT")
    )
    # 64-bit, not int32: EIP-155 chain ids aren't bounded by 2**31 (Palm is
    # 11297108109), and an overflow here would raise on the row that records
    # intent *before* the broadcast — the one write that must not fail.
    chain_id: Mapped[int] = mapped_column(BigInteger)
    to_address: Mapped[str] = mapped_column(String(42))
    recipient_label: Mapped[str] = mapped_column(String(64))
    value_wei: Mapped[Decimal] = mapped_column(Wei)
    gas_limit: Mapped[int | None] = mapped_column(BigInteger)
    max_fee_per_gas_wei: Mapped[Decimal | None] = mapped_column(Wei)
    max_priority_fee_per_gas_wei: Mapped[Decimal | None] = mapped_column(Wei)
    nonce: Mapped[int | None] = mapped_column(BigInteger)
    # "intended" -> "broadcast" -> "confirmed" | "failed"; a row never leaves
    # "intended" without us having tried, which is the point of writing it first.
    status: Mapped[str] = mapped_column(String(16))
    tx_hash: Mapped[str | None] = mapped_column(String(66))
    block_number: Mapped[int | None] = mapped_column(BigInteger)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    broadcast_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
