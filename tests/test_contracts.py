"""T2 contract guarantees: the Proposal/EvidenceRef shape and the canonical hash.

These tests pin the properties the rest of the system leans on — immutability,
exact money, tz-aware time, traceable evidence, and a content hash that depends
on meaning rather than field order.
"""

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from hodlin_contracts import (
    SCHEMA_VERSION,
    EvidenceRef,
    Proposal,
    canonical_hash,
    canonical_json,
)
from pydantic import ValidationError


def _evidence(**overrides: object) -> EvidenceRef:
    base: dict[str, object] = {
        "kind": "news",
        "source": "coindesk",
        "ref": "https://example.test/article",
        "observed_at": datetime(2026, 6, 28, 12, 0, tzinfo=UTC),
    }
    base.update(overrides)
    return EvidenceRef(**base)  # type: ignore[arg-type]


def _proposal(**overrides: object) -> Proposal:
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
    return Proposal(**base)  # type: ignore[arg-type]


def test_valid_proposal_round_trips() -> None:
    proposal = _proposal()
    assert proposal.schema_version == SCHEMA_VERSION == "1.0"
    assert proposal.amount == Decimal("0.5")


def test_proposal_is_frozen() -> None:
    proposal = _proposal()
    with pytest.raises(ValidationError):
        proposal.asset = "ETH"


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        _proposal(destination_address="0xdeadbeef")


def test_money_rejects_float() -> None:
    with pytest.raises(ValidationError):
        _proposal(amount=0.5)


def test_money_accepts_string_exactly() -> None:
    proposal = _proposal(amount="0.1")
    assert proposal.amount == Decimal("0.1")


def test_negative_amount_rejected() -> None:
    with pytest.raises(ValidationError):
        _proposal(amount=Decimal("-1"))


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        _proposal(created_at=datetime(2026, 6, 28, 12, 5))


def test_at_least_one_evidence_required() -> None:
    with pytest.raises(ValidationError):
        _proposal(evidence=())


def test_empty_required_strings_rejected() -> None:
    with pytest.raises(ValidationError):
        _proposal(asset="")
    with pytest.raises(ValidationError):
        _proposal(recipient_label="")


def test_canonical_hash_stable_across_construction_order() -> None:
    pid = uuid4()
    ts = datetime(2026, 6, 28, 12, 5, tzinfo=UTC)
    ev = _evidence()
    first = _proposal(proposal_id=pid, created_at=ts, evidence=(ev,))
    # Same meaning, fields supplied in a different order at construction.
    second = Proposal(
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


def test_canonical_hash_changes_with_meaning() -> None:
    pid = uuid4()
    ts = datetime(2026, 6, 28, 12, 5, tzinfo=UTC)
    a = _proposal(proposal_id=pid, created_at=ts, amount=Decimal("0.5"))
    b = _proposal(proposal_id=pid, created_at=ts, amount=Decimal("0.6"))
    assert canonical_hash(a) != canonical_hash(b)


def test_canonical_hash_ignores_decimal_scale() -> None:
    # 0.5 and 0.50 are equal in meaning; their proposals must hash identically.
    pid = uuid4()
    ts = datetime(2026, 6, 28, 12, 5, tzinfo=UTC)
    a = _proposal(proposal_id=pid, created_at=ts, amount=Decimal("0.5"))
    b = _proposal(proposal_id=pid, created_at=ts, amount="0.50")
    assert a == b
    assert canonical_hash(a) == canonical_hash(b)


def test_canonical_hash_ignores_whole_number_scale() -> None:
    # Whole amounts must not leak scientific notation ("1E+2") into the hash.
    pid = uuid4()
    ts = datetime(2026, 6, 28, 12, 5, tzinfo=UTC)
    a = _proposal(proposal_id=pid, created_at=ts, amount=Decimal("100"))
    b = _proposal(proposal_id=pid, created_at=ts, amount="100.00")
    assert canonical_hash(a) == canonical_hash(b)
    assert "E" not in canonical_json(a)


def test_large_whole_amount_normalizes_without_crash() -> None:
    # Digit count beyond the default decimal precision must not raise, and
    # must stay plain (no scientific notation) in the canonical form.
    pid = uuid4()
    ts = datetime(2026, 6, 28, 12, 5, tzinfo=UTC)
    a = _proposal(proposal_id=pid, created_at=ts, amount="1E+30")
    b = _proposal(proposal_id=pid, created_at=ts, amount="1" + "0" * 30)
    assert canonical_hash(a) == canonical_hash(b)
    assert "E" not in canonical_json(a)


def test_negative_zero_hashes_as_zero() -> None:
    pid = uuid4()
    ts = datetime(2026, 6, 28, 12, 5, tzinfo=UTC)
    a = _proposal(proposal_id=pid, created_at=ts, amount=Decimal("-0"))
    b = _proposal(proposal_id=pid, created_at=ts, amount=Decimal("0"))
    assert canonical_hash(a) == canonical_hash(b)


def test_canonical_hash_ignores_timezone_offset() -> None:
    # The same instant in two timezones must hash identically.
    pid = uuid4()
    utc_ts = datetime(2026, 6, 28, 12, 5, tzinfo=UTC)
    offset_ts = datetime(2026, 6, 28, 13, 5, tzinfo=timezone(timedelta(hours=1)))
    a = _proposal(proposal_id=pid, created_at=utc_ts)
    b = _proposal(proposal_id=pid, created_at=offset_ts)
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
