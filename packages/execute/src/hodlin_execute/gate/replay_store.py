"""The replay store: single-use, enforced by the database (T14, D15).

T13's token is stateless, so it can only say *how long* an approval is good for.
Inside that window the same token verifies as many times as it is presented —
which for a transfer means the same approval could move money twice. "At most
once" is a different property and needs state: this module.

**The primitive.** One conditional UPDATE, never a read followed by a write::

    UPDATE auth_tokens SET consumed_at = :now
     WHERE jti = :jti AND consumed_at IS NULL AND expires_at > :now
    RETURNING ...

Postgres evaluates that atomically, so two processes racing on the same ``jti``
cannot both find it unconsumed: the loser blocks on the row lock, re-reads after
the winner commits, matches zero rows, and is told so. A ``SELECT`` followed by an
``UPDATE`` would leave exactly the window this exists to close, and no amount of
application-level care closes it — this is the same reasoning as T9's claim on
``notified_at``, one layer more consequential.

**Deliberately the opposite trade-off from T9.** There, delivery is at-*least*-once:
a duplicate alert beats a missing one, so the claim commits after a successful
send. Here the token is consumed *before* the transaction is built (T17), making
it at-*most*-once: a transfer the human has to retry beats a transfer that
happened twice. Same compare-and-set, opposite direction, because the cost of the
duplicate is what changed.

**No I/O decisions hide in here.** The store never commits — the caller owns the
transaction boundary, because consuming a token and recording what was done with
it belong to one unit of work (T16). ``now`` is a parameter for the same reason it
is in ``token.py``.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from sqlalchemy import case, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from hodlin_execute.gate.token import TokenClaims
from hodlin_execute.store import tables


class ConsumedReason(StrEnum):
    """Why a token stopped being live. Stored on the row, so an operator can tell
    "spent on a transfer" from "replaced by a later approval" months later."""

    SPENT = "spent"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


class RegisterRejection(StrEnum):
    LIVE_TOKEN_EXISTS = "live_token_exists"
    DUPLICATE_JTI = "duplicate_jti"
    UNKNOWN_OPERATOR = "unknown_operator"
    ALREADY_EXPIRED = "already_expired"


class ConsumeRejection(StrEnum):
    UNKNOWN_JTI = "unknown_jti"
    #: Spent — the replay case, and the only one that suggests an attack.
    ALREADY_CONSUMED = "already_consumed"
    #: Retired by a later approval of the same proposal. Ordinary operation: a
    #: human clicking a stale approval link should not read as a replay attempt,
    #: because that difference is what decides whether anyone investigates.
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class Registered:
    jti: UUID


@dataclass(frozen=True, slots=True)
class RegisterRejected:
    reason: RegisterRejection


@dataclass(frozen=True, slots=True)
class Consumed:
    """The caller now holds the only claim on this token, and may act on it."""

    jti: UUID
    proposal_hash: str
    operator_id: int


@dataclass(frozen=True, slots=True)
class ConsumeRejected:
    reason: ConsumeRejection


type RegisterResult = Registered | RegisterRejected
type ConsumeResult = Consumed | ConsumeRejected


class ReplayStore:
    """Records minted tokens and consumes them exactly once."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def register(self, claims: TokenClaims, *, now: datetime) -> RegisterResult:
        """Record a freshly minted token so it can be consumed later.

        The row is the token's only existence as far as this domain is concerned:
        a token that verifies but was never registered has nothing to consume, and
        is therefore unusable — which is the property that makes a leaked secret
        insufficient to spend anything on its own.

        **Attribution is resolved from the claims, not accepted as an argument.**
        ``token.py`` puts ``subject`` inside the MAC on the grounds that "an audit
        trail that can be edited by the caller is not an audit trail" — and an
        ``operator_id`` parameter would have handed that same editing power back,
        one layer down. ``Consumed.operator_id`` is what T17 attributes a transfer
        to, so a caller mixing up the acting and approving operator would move
        money against the wrong human with nothing to detect it. Looking the
        operator up by the MAC-protected subject makes that unrepresentable rather
        than merely discouraged.

        The insert is wrapped in a SAVEPOINT so a constraint violation can be
        turned into a value without discarding the caller's transaction. T16
        registers a token in the middle of a larger unit of work (authorize →
        mint → audit), and losing the audit row because the mint collided would be
        the wrong trade.
        """
        if now.tzinfo is None:
            raise ValueError("`now` must be timezone-aware")
        # A token that is already dead on arrival would still occupy the one live
        # slot for its proposal — unconsumable, and blocking every later approval
        # until someone thought to supersede it. A backdated ``expires_at`` is a
        # clock or TTL bug upstream, so it gets refused here rather than stored.
        if claims.expires_at <= now:
            return RegisterRejected(RegisterRejection.ALREADY_EXPIRED)

        operator_id = (
            await self._session.execute(
                select(tables.Operator.id).where(tables.Operator.subject == claims.subject)
            )
        ).scalar_one_or_none()
        if operator_id is None:
            return RegisterRejected(RegisterRejection.UNKNOWN_OPERATOR)

        row = tables.AuthToken(
            jti=claims.jti,
            proposal_hash=claims.proposal_hash,
            operator_id=operator_id,
            issued_at=claims.issued_at,
            expires_at=claims.expires_at,
        )
        try:
            async with self._session.begin_nested():
                self._session.add(row)
                await self._session.flush()
        except IntegrityError as exc:
            return RegisterRejected(_classify_integrity_error(exc))
        return Registered(claims.jti)

    async def consume(self, jti: UUID, *, now: datetime) -> ConsumeResult:
        """Claim the token, atomically, or explain why it can't be claimed.

        This single statement *is* the security decision. Everything after it is
        bookkeeping: a token either transitioned from live to consumed in this
        statement, or it did not.
        """
        if now.tzinfo is None:
            raise ValueError("`now` must be timezone-aware")

        stmt = (
            update(tables.AuthToken)
            .where(
                tables.AuthToken.jti == jti,
                tables.AuthToken.consumed_at.is_(None),
                tables.AuthToken.expires_at > now,
            )
            .values(consumed_at=now, consumed_reason=ConsumedReason.SPENT)
            .returning(
                tables.AuthToken.jti,
                tables.AuthToken.proposal_hash,
                tables.AuthToken.operator_id,
            )
        )
        claimed = (await self._session.execute(stmt)).one_or_none()
        if claimed is not None:
            return Consumed(
                jti=claimed.jti,
                proposal_hash=claimed.proposal_hash,
                operator_id=claimed.operator_id,
            )
        return ConsumeRejected(await self._why_not_claimable(jti, now=now))

    async def supersede_live(self, proposal_hash: str, *, now: datetime) -> int:
        """Retire any live token for this proposal, returning how many were retired.

        Re-approval has to go through here first, because the schema permits only
        one live token per proposal hash — deliberately, so "approved twice, two
        spendable tokens" is unrepresentable rather than merely avoided.

        Note that a token whose window has *passed* is still live to that index:
        the predicate can only test ``consumed_at``, since a partial index cannot
        reference ``now()``. So this also retires expired-but-unconsumed rows, with
        the reason recorded as ``expired`` rather than ``superseded`` — otherwise a
        proposal whose token was never used would become permanently unapprovable,
        which is the failure the T12 review turned up while reading the index.
        """
        if now.tzinfo is None:
            raise ValueError("`now` must be timezone-aware")

        stmt = (
            update(tables.AuthToken)
            .where(
                tables.AuthToken.proposal_hash == proposal_hash,
                tables.AuthToken.consumed_at.is_(None),
            )
            .values(
                consumed_at=now,
                # One statement retires both kinds, and records which it was: an
                # unused token that simply timed out is a different story from one
                # replaced by a later decision, and the audit trail should say so.
                consumed_reason=case(
                    (tables.AuthToken.expires_at <= now, ConsumedReason.EXPIRED.value),
                    else_=ConsumedReason.SUPERSEDED.value,
                ),
            )
            .returning(tables.AuthToken.jti)
        )
        return len((await self._session.execute(stmt)).all())

    async def _why_not_claimable(self, jti: UUID, *, now: datetime) -> ConsumeRejection:
        """Label a lost race, after the fact.

        Read-then-write is the wrong shape for a *decision*, but this is not a
        decision — the CAS above already refused, and nothing here can turn that
        refusal into permission. It exists so the audit trail says "replayed"
        rather than "no", because those call for different human reactions.
        """
        stmt = select(
            tables.AuthToken.consumed_at,
            tables.AuthToken.consumed_reason,
            tables.AuthToken.expires_at,
        ).where(tables.AuthToken.jti == jti)
        row = (await self._session.execute(stmt)).one_or_none()
        if row is None:
            return ConsumeRejection.UNKNOWN_JTI
        if row.consumed_at is not None:
            # Why it stopped being live decides whether a human should care. A
            # token retired by a later approval is someone clicking a stale link;
            # a spent one presented again is the replay this module exists to
            # stop. Reporting both as "already consumed" would send an operator
            # hunting for an attack that never happened.
            if row.consumed_reason == ConsumedReason.SUPERSEDED:
                return ConsumeRejection.SUPERSEDED
            if row.consumed_reason == ConsumedReason.EXPIRED:
                return ConsumeRejection.EXPIRED
            return ConsumeRejection.ALREADY_CONSUMED
        if row.expires_at <= now:
            return ConsumeRejection.EXPIRED
        # Live now, yet the CAS matched nothing — so the row did not exist when we
        # tried to claim it and was committed by a concurrent ``register`` in
        # between. (Not "lost a race to a consumer that rolled back": that case
        # makes the CAS *succeed*, which is what the rollback test demonstrates.)
        # At the moment of the claim there was nothing to claim.
        return ConsumeRejection.UNKNOWN_JTI


#: The constraints this module knows how to explain. Anything else is a bug, not
#: an outcome — see ``_classify_integrity_error``.
_KNOWN_CONSTRAINTS = {
    "uq_auth_tokens_one_live_per_proposal": RegisterRejection.LIVE_TOKEN_EXISTS,
    "auth_tokens_pkey": RegisterRejection.DUPLICATE_JTI,
}


def _classify_integrity_error(exc: IntegrityError) -> RegisterRejection:
    """Turn a *known* constraint violation into a value, and re-raise the rest.

    A catch-all ``else`` was the bug here. ``jti`` is a fresh uuid4, so a real
    primary-key collision is effectively impossible — which means that in
    production a reported ``DUPLICATE_JTI`` would almost always be something else
    entirely (a missing operator row, a NOT NULL violation, a constraint added
    next year) quietly relabelled. T16 would then write a plausible "duplicate
    jti" into the approval trail for what is actually a referential-integrity bug
    on the money path, and nothing would ever surface it.

    Matching on the constraint *name* rather than the message text: the name is
    part of the schema this repo owns and the migration tests assert, whereas the
    message is Postgres's to reword between versions.
    """
    constraint = getattr(getattr(exc.orig, "__cause__", None), "constraint_name", None)
    known = _KNOWN_CONSTRAINTS.get(constraint) if isinstance(constraint, str) else None
    if known is None:
        raise exc
    return known
