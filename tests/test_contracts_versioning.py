"""T12: 1.1 alongside an untouched 1.0 — and why that is the only safe way (D32).

The canonical hash covers every serialized field, so a hashed schema cannot be
edited in place: adding one field would change the digest of proposals that were
already hashed, and any token minted over such a digest would stop verifying. So
1.0 is frozen forever and 1.1 is a separate document.

The load-bearing test here is the first one. It pins 1.0's digest against a
committed fixture, which is the only thing that can *prove* the claim "old
proposals still hash the same" — a claim no amount of careful reading enforces.
"""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, get_args

import pytest
from hodlin_contracts import (
    SCHEMA_VERSION,
    EvidenceRef,
    ProposalV1_0,
    ProposalV1_1,
    canonical_hash,
    is_expired,
    parse_proposal,
)
from pydantic import ValidationError

_FIXTURES = Path(__file__).parent / "fixtures"

#: Pinned 2026-08-21 from the 1.0 model exactly as M1 shipped it. If a change to
#: the codebase moves this digest, then 1.0 has been mutated and every historical
#: hash and token minted over one is invalid — the code is the bug, not this
#: constant. A new shape belongs in a new version.
PROPOSAL_V1_0_DIGEST = "9dbaa8bdc4719b423311c0a90447a44fb3973ad10a7733941b057f65845a0803"

#: And 1.1's, pinned the day it shipped rather than the day it becomes expensive.
#: 1.1 is the version this codebase *produces*, so from T13 on its digests are the
#: ones tokens are minted over — at which point it is exactly as frozen as 1.0.
#: Pinning it now means "add a field to 1.1" fails a test instead of quietly
#: invalidating stored hashes; without this, doing so breaks nothing in the suite.
PROPOSAL_V1_1_DIGEST = "94c9ff9d41c074e244a2a9b1bd835b541fc51843ffe75e7f7b201fe66a3d0a68"

_CREATED_AT = datetime(2026, 6, 28, 12, 5, tzinfo=UTC)
_OBSERVED_AT = datetime(2026, 6, 28, 12, 0, tzinfo=UTC)


def _evidence() -> dict[str, Any]:
    return {
        "kind": "anomaly",
        "source": "hodlin",
        "ref": "BTC-USD@2026-06-28T12:00:00Z",
        "observed_at": _OBSERVED_AT,
    }


def _business_fields(**overrides: Any) -> dict[str, Any]:
    """The fields both versions share — deliberately identical, so the only
    difference between a 1.0 and a 1.1 built from these is the version itself."""
    base: dict[str, Any] = {
        "proposal_id": "6f1e9c74-2f1b-4d5e-8a3c-9b0d7e5f4a21",
        "asset": "BTC",
        "action": "buy",
        "amount": Decimal("0.5"),
        "recipient_label": "cold-wallet",
        "reasoning": "anomalous volume spike",
        "evidence": (_evidence(),),
        "created_at": _CREATED_AT,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("fixture", "expected_class", "digest"),
    [
        ("proposal_v1_0.json", ProposalV1_0, PROPOSAL_V1_0_DIGEST),
        ("proposal_v1_1.json", ProposalV1_1, PROPOSAL_V1_1_DIGEST),
    ],
    ids=["1.0", "1.1"],
)
def test_a_committed_proposal_still_validates_and_hashes_identically(
    fixture: str, expected_class: type[ProposalV1_0] | type[ProposalV1_1], digest: str
) -> None:
    """The regression that gives D32 its teeth: a document captured on the day its
    version shipped must still parse, and must still produce the digest it produced
    then. Every *live* version needs this, not only the historical one — the moment
    a version's digests are stored, editing that version invalidates them.

    Both fixtures are realistic wire payloads — keys out of order, a ``+01:00``
    offset, a trailing-zero amount — so this also asserts that canonicalization,
    not the sender's formatting, is what fixes the digest.
    """
    payload = json.loads((_FIXTURES / fixture).read_text())

    proposal = parse_proposal(payload)

    assert isinstance(proposal, expected_class)
    assert canonical_hash(proposal) == digest


def test_the_same_business_fields_hash_differently_across_versions() -> None:
    """``schema_version`` is inside the hash, so a cross-version collision is
    impossible by construction: an approval of the 1.0 document can never be
    replayed as an approval of the 1.1 one, even though they say the same thing
    about the same asset for the same amount."""
    fields = _business_fields()

    v1_0 = ProposalV1_0(**fields)
    v1_1 = ProposalV1_1(**fields)

    assert v1_0.model_dump(exclude={"schema_version"}) == v1_1.model_dump(
        exclude={"schema_version", "valid_until"}
    )
    assert canonical_hash(v1_0) != canonical_hash(v1_1)


def test_transfer_is_expressible_only_in_1_1() -> None:
    """The point of the bump. ``transfer`` is the action that moves value, so it
    arrives with the version that the execute gate understands — and 1.0 keeps
    rejecting it, which is what stops an old-shaped proposal from smuggling one
    in."""
    assert ProposalV1_1(**_business_fields(action="transfer")).action == "transfer"

    with pytest.raises(ValidationError):
        ProposalV1_0(**_business_fields(action="transfer"))


def test_a_1_1_payload_is_not_readable_as_1_0() -> None:
    """``extra="forbid"`` doing real work: 1.1's own field is unknown to 1.0, so
    a newer document cannot be silently narrowed into an older class (which would
    hash as something the sender never signed).

    The version tag is *relabelled* to "1.0" first, on purpose. Left as "1.1" it
    fails on the ``Literal["1.0"]`` mismatch alone, and the assertion would keep
    passing even if cross-version narrowing became possible — a test that proves
    something weaker than it claims. Relabelling is also the realistic attack: a
    caller who wants a 1.1 payload treated as 1.0 would obviously edit the tag.
    """
    payload = ProposalV1_1(**_business_fields(valid_until=_CREATED_AT + timedelta(hours=1)))
    relabelled = {**payload.model_dump(mode="json"), "schema_version": "1.0"}

    with pytest.raises(ValidationError, match="valid_until"):
        ProposalV1_0(**relabelled)


class TestParseProposal:
    """Dispatch on the version *in the payload*, never on a guess."""

    def test_each_version_parses_to_its_own_class(self) -> None:
        for model in (ProposalV1_0, ProposalV1_1):
            payload = model(**_business_fields()).model_dump(mode="json")
            assert type(parse_proposal(payload)) is model

    def test_an_unknown_version_is_rejected(self) -> None:
        """Not "fall back to the newest" — a version this build has never seen
        may mean anything, and hashing it as 1.1 would produce a digest for a
        document we didn't actually understand."""
        payload = ProposalV1_1(**_business_fields()).model_dump(mode="json")

        with pytest.raises(ValidationError):
            parse_proposal({**payload, "schema_version": "2.0"})

    def test_a_missing_version_is_rejected(self) -> None:
        """Direct construction defaults the version because in-process we know
        what we are building; a payload from outside has to say. Guessing here is
        how a 1.0 document gets read as 1.1."""
        payload = ProposalV1_1(**_business_fields()).model_dump(mode="json")
        del payload["schema_version"]

        with pytest.raises(ValidationError):
            parse_proposal(payload)


def test_a_future_version_with_a_deadline_is_honoured_without_touching_is_expired() -> None:
    """The failure this guards against is silent and lands on the money path.

    ``is_expired`` reads ``valid_until`` structurally rather than matching known
    classes, so a version that doesn't exist yet is handled the moment it does. An
    ``isinstance`` chain would type-check fine after the union was widened, return
    ``False`` for a proposal whose deadline passed months ago, and let the gate
    approve it believing it had asked. This stand-in *is* the next version as far as
    the helper is concerned.
    """

    class ProposalV1_2Stub(ProposalV1_1):
        schema_version: Literal["1.2"] = "1.2"  # type: ignore[assignment]  # a stand-in for the next version

    stale = ProposalV1_2Stub(**_business_fields(valid_until=_CREATED_AT + timedelta(minutes=5)))

    assert is_expired(stale, at=_CREATED_AT + timedelta(hours=1))
    assert not is_expired(stale, at=_CREATED_AT + timedelta(minutes=1))


def test_the_allowed_evidence_kinds_are_pinned() -> None:
    """``EvidenceRef`` is shared by every proposal version, so widening ``kind``
    widens what a *frozen* 1.0 document may say — without changing any existing
    digest, which is what makes it easy to miss. This test doesn't forbid a new
    kind; it forces the "does this need a new proposal version?" conversation
    before one lands."""
    kind_field = EvidenceRef.model_fields["kind"]

    assert get_args(kind_field.annotation) == ("anomaly", "news", "sentiment", "price")


class TestAmountFloor:
    """1.1 adds the action that moves value, so it adds the floor that makes a
    value-moving proposal meaningful."""

    @pytest.mark.parametrize("action", ["buy", "sell", "transfer"])
    def test_a_zero_amount_value_moving_action_is_rejected(self, action: str) -> None:
        """A zero-amount transfer burns gas to do nothing, and is the shape a
        prompt-injected recommend domain emits when it's flailing."""
        with pytest.raises(ValidationError, match="greater than zero"):
            ProposalV1_1(**_business_fields(action=action, amount=Decimal("0")))

    @pytest.mark.parametrize("action", ["hold", "alert"])
    def test_actions_that_move_nothing_may_carry_zero(self, action: str) -> None:
        assert ProposalV1_1(**_business_fields(action=action, amount=Decimal("0"))).amount == 0

    def test_1_0_keeps_its_own_rules(self) -> None:
        """Tightening validation on a frozen version could make a stored proposal
        unparseable — the D32 mutation in another guise. 1.0 still accepts what it
        always accepted; the gate is free to refuse it."""
        assert ProposalV1_0(**_business_fields(action="buy", amount=Decimal("0"))).amount == 0


class TestValidUntil:
    """Freshness is asked at the gate, not enforced at parse time."""

    def test_a_past_deadline_still_parses(self) -> None:
        """A historical proposal must stay parseable forever — re-reading a
        stored proposal is not the same act as accepting a new one. If parsing
        rejected stale proposals, the audit trail would become unreadable with
        the passage of time, and a stored digest unverifiable."""
        stale = ProposalV1_1(**_business_fields(valid_until=_CREATED_AT + timedelta(minutes=1)))

        assert stale.valid_until == _CREATED_AT + timedelta(minutes=1)
        assert is_expired(stale, at=datetime(2026, 8, 21, tzinfo=UTC))

    def test_a_deadline_in_the_future_is_not_expired(self) -> None:
        fresh = ProposalV1_1(**_business_fields(valid_until=_CREATED_AT + timedelta(hours=2)))

        assert not is_expired(fresh, at=_CREATED_AT + timedelta(hours=1))

    def test_no_deadline_never_expires(self) -> None:
        assert not is_expired(ProposalV1_1(**_business_fields()), at=_CREATED_AT)

    def test_1_0_has_no_deadline_of_its_own(self) -> None:
        """Not a loophole: the gate's short-lived, single-use token still bounds
        a 1.0 approval. ``valid_until`` bounds the *recommendation*, the token
        bounds the *authorization*, and they are different clocks."""
        assert not is_expired(
            ProposalV1_0(**_business_fields()), at=datetime(2030, 1, 1, tzinfo=UTC)
        )

    def test_a_deadline_at_or_before_creation_is_rejected(self) -> None:
        """A static contradiction between two of the proposal's own fields — it
        was never actionable. Safe to reject at parse time precisely because the
        verdict can't change later: no stored proposal becomes unparseable."""
        for offset in (timedelta(0), timedelta(seconds=-1)):
            with pytest.raises(ValidationError):
                ProposalV1_1(**_business_fields(valid_until=_CREATED_AT + offset))

    def test_comparing_against_a_naive_now_is_refused(self) -> None:
        """A naive datetime would silently compare against an unknown zone (or
        raise deep inside), and this decision gates money."""
        proposal = ProposalV1_1(**_business_fields(valid_until=_CREATED_AT + timedelta(hours=1)))

        with pytest.raises(ValueError, match="timezone-aware"):
            is_expired(proposal, at=datetime(2026, 6, 28, 13, 0))

    def test_valid_until_is_inside_the_hash(self) -> None:
        """It changes what was authorized, so it must change the digest."""
        without = ProposalV1_1(**_business_fields())
        with_deadline = ProposalV1_1(
            **_business_fields(valid_until=_CREATED_AT + timedelta(hours=1))
        )

        assert canonical_hash(without) != canonical_hash(with_deadline)


def test_the_current_version_constant_matches_the_class_it_names() -> None:
    """One place says what this codebase produces, and it has to agree with the
    class that produces it — otherwise a consumer trusting ``SCHEMA_VERSION``
    reads the wrong contract."""
    assert SCHEMA_VERSION == ProposalV1_1(**_business_fields()).schema_version
