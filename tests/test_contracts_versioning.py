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
from typing import Any

import pytest
from hodlin_contracts import (
    SCHEMA_VERSION,
    ProposalV1_0,
    ProposalV1_1,
    canonical_hash,
    is_expired,
    parse_proposal,
)
from pydantic import ValidationError

_FIXTURE = Path(__file__).parent / "fixtures" / "proposal_v1_0.json"

#: Pinned 2026-08-21 from the 1.0 model exactly as M1 shipped it. If a change to
#: the codebase moves this digest, then 1.0 has been mutated and every historical
#: hash and token minted over one is invalid — the code is the bug, not this
#: constant. A new shape belongs in a new version.
PROPOSAL_V1_0_DIGEST = "9dbaa8bdc4719b423311c0a90447a44fb3973ad10a7733941b057f65845a0803"

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


def test_the_committed_1_0_proposal_still_validates_and_hashes_identically() -> None:
    """The regression that gives D32 its teeth: a 1.0 document captured before
    1.1 existed must still parse, and must still produce the digest it produced
    then. Note the fixture is a realistic wire payload — keys out of order, a
    ``+01:00`` offset, a trailing-zero amount — so this also asserts that
    canonicalization, not the sender's formatting, is what fixes the digest."""
    payload = json.loads(_FIXTURE.read_text())

    proposal = parse_proposal(payload)

    assert isinstance(proposal, ProposalV1_0)
    assert proposal.schema_version == "1.0"
    assert canonical_hash(proposal) == PROPOSAL_V1_0_DIGEST


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
    hash as something the sender never signed)."""
    payload = ProposalV1_1(**_business_fields(valid_until=_CREATED_AT + timedelta(hours=1)))

    with pytest.raises(ValidationError):
        ProposalV1_0(**payload.model_dump(mode="json"))


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
