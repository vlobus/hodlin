"""T2 contract guarantees: the proposal/evidence shape and the canonical hash.

These tests pin the properties the rest of the system leans on — immutability,
exact money, tz-aware time, traceable evidence, and a content hash that depends
on meaning rather than field order.

Since T12 there are two proposal versions, so every guarantee that is *not*
version-specific runs against **both** of them: a new version must not quietly
drop an invariant the old one had. What 1.1 adds, and how the two versions relate,
lives in ``test_contracts_versioning.py``.
"""

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from hodlin_contracts import (
    SCHEMA_VERSION,
    EvidenceRef,
    ProposalV1_0,
    ProposalV1_1,
    canonical_hash,
    canonical_json,
)
from pydantic import ValidationError

type ProposalModel = type[ProposalV1_0] | type[ProposalV1_1]

#: Both versions, for the invariants neither may lose.
BOTH_VERSIONS = pytest.mark.parametrize("model", [ProposalV1_0, ProposalV1_1], ids=["1.0", "1.1"])


def _evidence(**overrides: object) -> EvidenceRef:
    base: dict[str, object] = {
        "kind": "news",
        "source": "coindesk",
        "ref": "https://example.test/article",
        "observed_at": datetime(2026, 6, 28, 12, 0, tzinfo=UTC),
    }
    base.update(overrides)
    return EvidenceRef(**base)  # type: ignore[arg-type]


def _proposal(
    model: ProposalModel = ProposalV1_1, **overrides: object
) -> ProposalV1_0 | ProposalV1_1:
    base: dict[str, object] = {
        "proposal_id": uuid4(),
        "asset": "BTC",
        "action": "buy",
        "amount": Decimal("0.5"),
        "recipient_label": "cold-wallet",
        "reasoning": "anomalous volume spike",
        "evidence": (_evidence(),),
        "created_at": datetime(2026, 6, 28, 12, 5, tzinfo=UTC),
    }
    base.update(overrides)
    return model(**base)  # type: ignore[arg-type]


@BOTH_VERSIONS
def test_valid_proposal_round_trips(model: ProposalModel) -> None:
    proposal = _proposal(model)
    assert proposal.amount == Decimal("0.5")


def test_the_current_version_is_what_gets_produced() -> None:
    assert SCHEMA_VERSION == "1.1"
    assert _proposal(ProposalV1_1).schema_version == SCHEMA_VERSION
    assert _proposal(ProposalV1_0).schema_version == "1.0"


@BOTH_VERSIONS
def test_proposal_is_frozen(model: ProposalModel) -> None:
    proposal = _proposal(
        model,
    )
    with pytest.raises(ValidationError):
        proposal.asset = "ETH"


@BOTH_VERSIONS
def test_unknown_fields_rejected(model: ProposalModel) -> None:
    with pytest.raises(ValidationError):
        _proposal(model, destination_address="0xdeadbeef")


@BOTH_VERSIONS
def test_money_rejects_float(model: ProposalModel) -> None:
    with pytest.raises(ValidationError):
        _proposal(model, amount=0.5)


@BOTH_VERSIONS
def test_money_accepts_string_exactly(model: ProposalModel) -> None:
    proposal = _proposal(model, amount="0.1")
    assert proposal.amount == Decimal("0.1")


@BOTH_VERSIONS
def test_negative_amount_rejected(model: ProposalModel) -> None:
    with pytest.raises(ValidationError):
        _proposal(model, amount=Decimal("-1"))


@BOTH_VERSIONS
def test_naive_datetime_rejected(model: ProposalModel) -> None:
    with pytest.raises(ValidationError):
        _proposal(model, created_at=datetime(2026, 6, 28, 12, 5))


@BOTH_VERSIONS
def test_at_least_one_evidence_required(model: ProposalModel) -> None:
    with pytest.raises(ValidationError):
        _proposal(model, evidence=())


@BOTH_VERSIONS
def test_empty_required_strings_rejected(model: ProposalModel) -> None:
    with pytest.raises(ValidationError):
        _proposal(model, asset="")
    with pytest.raises(ValidationError):
        _proposal(model, recipient_label="")


@BOTH_VERSIONS
def test_canonical_hash_stable_across_construction_order(model: ProposalModel) -> None:
    pid = uuid4()
    ts = datetime(2026, 6, 28, 12, 5, tzinfo=UTC)
    ev = _evidence()
    first = _proposal(model, proposal_id=pid, created_at=ts, evidence=(ev,))
    # Same meaning, fields supplied in a different order at construction.
    second = model(
        created_at=ts,
        evidence=(ev,),
        reasoning="anomalous volume spike",
        recipient_label="cold-wallet",
        amount=Decimal("0.5"),
        action="buy",
        asset="BTC",
        proposal_id=pid,
    )
    assert canonical_hash(first) == canonical_hash(second)


@BOTH_VERSIONS
def test_canonical_hash_changes_with_meaning(model: ProposalModel) -> None:
    pid = uuid4()
    ts = datetime(2026, 6, 28, 12, 5, tzinfo=UTC)
    a = _proposal(model, proposal_id=pid, created_at=ts, amount=Decimal("0.5"))
    b = _proposal(model, proposal_id=pid, created_at=ts, amount=Decimal("0.6"))
    assert canonical_hash(a) != canonical_hash(b)


@BOTH_VERSIONS
def test_canonical_hash_ignores_decimal_scale(model: ProposalModel) -> None:
    # 0.5 and 0.50 are equal in meaning; their proposals must hash identically.
    pid = uuid4()
    ts = datetime(2026, 6, 28, 12, 5, tzinfo=UTC)
    a = _proposal(model, proposal_id=pid, created_at=ts, amount=Decimal("0.5"))
    b = _proposal(model, proposal_id=pid, created_at=ts, amount="0.50")
    assert a == b
    assert canonical_hash(a) == canonical_hash(b)


@BOTH_VERSIONS
def test_canonical_hash_ignores_whole_number_scale(model: ProposalModel) -> None:
    # Whole amounts must not leak scientific notation ("1E+2") into the hash.
    pid = uuid4()
    ts = datetime(2026, 6, 28, 12, 5, tzinfo=UTC)
    a = _proposal(model, proposal_id=pid, created_at=ts, amount=Decimal("100"))
    b = _proposal(model, proposal_id=pid, created_at=ts, amount="100.00")
    assert canonical_hash(a) == canonical_hash(b)
    assert "E" not in canonical_json(a)


@BOTH_VERSIONS
def test_large_whole_amount_normalizes_without_crash(model: ProposalModel) -> None:
    # Digit count beyond the default decimal precision must not raise, and
    # must stay plain (no scientific notation) in the canonical form.
    pid = uuid4()
    ts = datetime(2026, 6, 28, 12, 5, tzinfo=UTC)
    a = _proposal(model, proposal_id=pid, created_at=ts, amount="1E+30")
    b = _proposal(model, proposal_id=pid, created_at=ts, amount="1" + "0" * 30)
    assert canonical_hash(a) == canonical_hash(b)
    assert "E" not in canonical_json(a)


@BOTH_VERSIONS
def test_negative_zero_hashes_as_zero(model: ProposalModel) -> None:
    pid = uuid4()
    ts = datetime(2026, 6, 28, 12, 5, tzinfo=UTC)
    a = _proposal(model, proposal_id=pid, created_at=ts, amount=Decimal("-0"))
    b = _proposal(model, proposal_id=pid, created_at=ts, amount=Decimal("0"))
    assert canonical_hash(a) == canonical_hash(b)


@BOTH_VERSIONS
def test_canonical_hash_ignores_timezone_offset(model: ProposalModel) -> None:
    # The same instant in two timezones must hash identically.
    pid = uuid4()
    utc_ts = datetime(2026, 6, 28, 12, 5, tzinfo=UTC)
    offset_ts = datetime(2026, 6, 28, 13, 5, tzinfo=timezone(timedelta(hours=1)))
    a = _proposal(model, proposal_id=pid, created_at=utc_ts)
    b = _proposal(model, proposal_id=pid, created_at=offset_ts)
    assert a == b
    assert canonical_hash(a) == canonical_hash(b)


def test_evidence_ref_is_frozen() -> None:
    ev = _evidence()
    with pytest.raises(ValidationError):
        ev.source = "other"


def test_evidence_ref_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _evidence(extra="nope")


def test_whitespace_only_strings_rejected() -> None:
    with pytest.raises(ValidationError):
        _proposal(asset="   ")


def test_canonical_json_has_sorted_keys_and_no_whitespace() -> None:
    text = canonical_json(_proposal())
    assert ", " not in text and ": " not in text
    assert '"action"' in text
    # action precedes amount precedes asset -> keys are sorted.
    assert text.index('"action"') < text.index('"amount"') < text.index('"asset"')
