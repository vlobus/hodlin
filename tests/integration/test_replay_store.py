"""T14: single-use, proven against real Postgres with two concurrent sessions.

The claim this file has to earn is "the same approval cannot move money twice".
That claim is about *concurrency*, so it can only be made against a real database:
an in-memory double would happily let both callers win, and mocking the very
statement whose atomicity is the point would assert nothing at all.

The decisive test is `test_two_concurrent_consumers_and_exactly_one_wins`, which
runs the race deliberately — B's update is issued while A holds the row lock,
asserted to be *blocked*, and only then does A commit. Both halves matter: that
one caller wins, and that the other was made to wait rather than reading stale
state.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from hodlin_execute.gate.replay_store import (
    Consumed,
    ConsumedReason,
    ConsumeRejected,
    ConsumeRejection,
    Registered,
    RegisterRejected,
    RegisterRejection,
    ReplayStore,
)
from hodlin_execute.gate.token import TokenClaims
from hodlin_execute.store import tables
from hodlin_execute.store.db import create_session_factory
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

_NOW = datetime(2026, 8, 23, 12, 0, tzinfo=UTC)
_EXPIRES_AT = _NOW + timedelta(minutes=5)
_HASH = "9dbaa8bdc4719b423311c0a90447a44fb3973ad10a7733941b057f65845a0803"
_OTHER_HASH = "94c9ff9d41c074e244a2a9b1bd835b541fc51843ffe75e7f7b201fe66a3d0a68"


def _claims(**overrides: object) -> TokenClaims:
    base: dict[str, object] = {
        "proposal_hash": _HASH,
        "jti": uuid4(),
        "subject": "keycloak|operator-42",
        "amount": Decimal("0.25"),
        "recipient_label": "cold-wallet",
        "issued_at": _NOW,
        "expires_at": _EXPIRES_AT,
    }
    base.update(overrides)
    return TokenClaims(**base)  # type: ignore[arg-type]


@pytest.fixture
async def operator_id(execute_session: AsyncSession) -> int:
    """A real operator row: ``auth_tokens.operator_id`` is a RESTRICT foreign key,
    so a token can't exist without the human it is attributed to."""
    operator = tables.Operator(subject="keycloak|operator-42")
    execute_session.add(operator)
    await execute_session.flush()
    return operator.id


async def _consumed_row(session: AsyncSession, jti: UUID) -> tables.AuthToken:
    return (
        await session.execute(select(tables.AuthToken).where(tables.AuthToken.jti == jti))
    ).scalar_one()


class TestConsume:
    async def test_a_live_token_is_consumed_once_and_then_refused(
        self, execute_session: AsyncSession, operator_id: int
    ) -> None:
        """The whole property in one test: the second attempt is told it is a
        replay, not merely refused."""
        store = ReplayStore(execute_session)
        claims = _claims()
        assert isinstance(await store.register(claims, operator_id=operator_id), Registered)

        first = await store.consume(claims.jti, now=_NOW + timedelta(minutes=1))
        second = await store.consume(claims.jti, now=_NOW + timedelta(minutes=1))

        assert first == Consumed(jti=claims.jti, proposal_hash=_HASH, operator_id=operator_id)
        assert second == ConsumeRejected(ConsumeRejection.ALREADY_CONSUMED)

    async def test_consumption_is_recorded_with_its_reason(
        self, execute_session: AsyncSession, operator_id: int
    ) -> None:
        """The row is the audit trail, so *why* it stopped being live is part of
        the data, not something to infer from timestamps later."""
        store = ReplayStore(execute_session)
        claims = _claims()
        await store.register(claims, operator_id=operator_id)

        await store.consume(claims.jti, now=_NOW + timedelta(minutes=1))

        row = await _consumed_row(execute_session, claims.jti)
        assert row.consumed_at == _NOW + timedelta(minutes=1)
        assert row.consumed_reason == ConsumedReason.SPENT

    async def test_an_expired_token_cannot_be_consumed(
        self, execute_session: AsyncSession, operator_id: int
    ) -> None:
        """Defence in depth rather than duplication: T13's verifier already checks
        the window, but the store is the last thing standing before the transfer,
        and it should not depend on an earlier caller having asked."""
        store = ReplayStore(execute_session)
        claims = _claims()
        await store.register(claims, operator_id=operator_id)

        result = await store.consume(claims.jti, now=_EXPIRES_AT + timedelta(seconds=1))

        assert result == ConsumeRejected(ConsumeRejection.EXPIRED)

    async def test_an_unknown_jti_is_distinguishable_from_a_replay(
        self, execute_session: AsyncSession
    ) -> None:
        """A token that verifies but was never registered has nothing to consume —
        which is what makes a leaked secret insufficient on its own — and reads
        very differently in an audit log from a replay."""
        result = await ReplayStore(execute_session).consume(uuid4(), now=_NOW)

        assert result == ConsumeRejected(ConsumeRejection.UNKNOWN_JTI)

    async def test_a_naive_now_is_a_programming_error(self, execute_session: AsyncSession) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            await ReplayStore(execute_session).consume(uuid4(), now=datetime(2026, 8, 23, 12, 0))


class TestConcurrency:
    """The reason this file needs a real database."""

    async def test_two_concurrent_consumers_and_exactly_one_wins(
        self, execute_engine: AsyncEngine, execute_session: AsyncSession, operator_id: int
    ) -> None:
        """Run the race for real: two sessions, two transactions, one row.

        A consumes but does not commit, so it holds the row lock. B's identical
        statement is then issued and must *block* — asserted, because if B were
        able to proceed it would be reading state A is in the middle of changing,
        which is precisely the read-then-write window this design exists to close.
        When A commits, B re-evaluates its predicate under READ COMMITTED, matches
        zero rows, and reports a replay.
        """
        claims = _claims()
        await ReplayStore(execute_session).register(claims, operator_id=operator_id)
        await execute_session.commit()

        factory = create_session_factory(execute_engine)
        async with factory() as session_a, factory() as session_b:
            won_a = await ReplayStore(session_a).consume(claims.jti, now=_NOW)

            racing_b = asyncio.create_task(ReplayStore(session_b).consume(claims.jti, now=_NOW))
            await asyncio.sleep(0.2)
            assert not racing_b.done(), "B proceeded while A held the row lock"

            await session_a.commit()
            won_b = await asyncio.wait_for(racing_b, timeout=10)
            await session_b.commit()

        assert isinstance(won_a, Consumed)
        assert won_b == ConsumeRejected(ConsumeRejection.ALREADY_CONSUMED)

    async def test_a_rolled_back_consumer_leaves_the_token_spendable(
        self, execute_engine: AsyncEngine, execute_session: AsyncSession, operator_id: int
    ) -> None:
        """At-most-once is a property of *committed* work.

        If the transaction that consumed the token rolls back — the transfer never
        left the process — the token must still be spendable, or a crash between
        claim and broadcast would burn a human's approval with nothing to show for
        it. This is the difference between the token being consumed and the money
        having moved, and it's why T17 commits the claim and the intent together.
        """
        claims = _claims()
        await ReplayStore(execute_session).register(claims, operator_id=operator_id)
        await execute_session.commit()

        factory = create_session_factory(execute_engine)
        async with factory() as doomed:
            assert isinstance(await ReplayStore(doomed).consume(claims.jti, now=_NOW), Consumed)
            await doomed.rollback()

        async with factory() as retry:
            assert isinstance(await ReplayStore(retry).consume(claims.jti, now=_NOW), Consumed)
            await retry.commit()


class TestRegister:
    async def test_one_live_token_per_proposal(
        self, execute_session: AsyncSession, operator_id: int
    ) -> None:
        """The schema, not this code, is what makes "approved twice, two spendable
        tokens" unrepresentable — so what's asserted here is that the constraint
        surfaces as a *value* the caller can act on rather than an exception."""
        store = ReplayStore(execute_session)
        await store.register(_claims(), operator_id=operator_id)

        second = await store.register(_claims(), operator_id=operator_id)

        assert second == RegisterRejected(RegisterRejection.LIVE_TOKEN_EXISTS)

    async def test_a_different_proposal_may_have_its_own_live_token(
        self, execute_session: AsyncSession, operator_id: int
    ) -> None:
        """The constraint is per proposal, not global — two proposals awaiting
        execution at once is normal operation, not a conflict."""
        store = ReplayStore(execute_session)
        await store.register(_claims(), operator_id=operator_id)

        other = await store.register(_claims(proposal_hash=_OTHER_HASH), operator_id=operator_id)

        assert isinstance(other, Registered)

    async def test_a_reused_jti_is_refused(
        self, execute_session: AsyncSession, operator_id: int
    ) -> None:
        store = ReplayStore(execute_session)
        claims = _claims()
        await store.register(claims, operator_id=operator_id)
        await store.consume(claims.jti, now=_NOW)

        again = await store.register(
            _claims(jti=claims.jti, proposal_hash=_OTHER_HASH), operator_id=operator_id
        )

        assert again == RegisterRejected(RegisterRejection.DUPLICATE_JTI)

    async def test_a_rejected_register_does_not_poison_the_transaction(
        self, execute_session: AsyncSession, operator_id: int
    ) -> None:
        """The SAVEPOINT earning its place: T16 registers a token inside a larger
        unit of work, and a collision must not cost the audit row that records
        what happened."""
        store = ReplayStore(execute_session)
        await store.register(_claims(), operator_id=operator_id)

        assert isinstance(
            await store.register(_claims(), operator_id=operator_id), RegisterRejected
        )

        execute_session.add(tables.Operator(subject="keycloak|still-usable"))
        await execute_session.flush()
        await execute_session.commit()


class TestSupersede:
    async def test_re_approval_retires_the_live_token_first(
        self, execute_session: AsyncSession, operator_id: int
    ) -> None:
        store = ReplayStore(execute_session)
        first = _claims()
        await store.register(first, operator_id=operator_id)

        retired = await store.supersede_live(_HASH, now=_NOW + timedelta(minutes=1))
        second = await store.register(_claims(), operator_id=operator_id)

        assert retired == 1
        assert isinstance(second, Registered)
        row = await _consumed_row(execute_session, first.jti)
        assert row.consumed_reason == ConsumedReason.SUPERSEDED

    async def test_the_superseded_token_can_no_longer_be_spent(
        self, execute_session: AsyncSession, operator_id: int
    ) -> None:
        """Otherwise re-approval would *add* authority instead of replacing it."""
        store = ReplayStore(execute_session)
        first = _claims()
        await store.register(first, operator_id=operator_id)
        await store.supersede_live(_HASH, now=_NOW + timedelta(minutes=1))

        result = await store.consume(first.jti, now=_NOW + timedelta(minutes=2))

        assert result == ConsumeRejected(ConsumeRejection.ALREADY_CONSUMED)

    async def test_an_expired_unused_token_does_not_block_re_approval_forever(
        self, execute_session: AsyncSession, operator_id: int
    ) -> None:
        """The hole the T12 review found by reading the partial index: it can only
        test ``consumed_at``, so an expired-but-unconsumed token still occupies the
        one-live-token slot. Without this path a proposal whose token timed out
        unused would be permanently unapprovable — and the reason recorded is
        ``expired``, not ``superseded``, because nobody made a new decision.
        """
        store = ReplayStore(execute_session)
        abandoned = _claims()
        await store.register(abandoned, operator_id=operator_id)
        after_expiry = _EXPIRES_AT + timedelta(hours=1)

        assert await store.register(_claims(), operator_id=operator_id) == RegisterRejected(
            RegisterRejection.LIVE_TOKEN_EXISTS
        )
        retired = await store.supersede_live(_HASH, now=after_expiry)

        assert retired == 1
        row = await _consumed_row(execute_session, abandoned.jti)
        assert row.consumed_reason == ConsumedReason.EXPIRED
        assert isinstance(
            await store.register(
                _claims(issued_at=after_expiry, expires_at=after_expiry + timedelta(minutes=5)),
                operator_id=operator_id,
            ),
            Registered,
        )

    async def test_superseding_nothing_is_not_an_error(self, execute_session: AsyncSession) -> None:
        """First approval of a proposal is the common case; it must not need a
        special path."""
        assert await ReplayStore(execute_session).supersede_live(_HASH, now=_NOW) == 0
