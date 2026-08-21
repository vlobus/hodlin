"""T13: the authorization token — what it binds, and every way it must refuse.

The positive case is one test. The rest of this file is the interesting part,
because a verifier is only worth what it *rejects*: a token lifted onto a bigger
amount, onto a different recipient, onto a different proposal, minted with another
secret, replayed after expiry, or handed over as garbage.

Two properties of the suite itself are deliberate:

* the tamper cases are **derived from ``TokenClaims.model_fields``**, so adding a
  claim without a tamper test fails the suite instead of quietly shipping an
  unverified binding;
* nothing here reads the clock. ``now`` is an argument everywhere, so "expired"
  is asserted rather than waited for, and the suite behaves identically at any
  hour and in any timezone.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, ClassVar
from uuid import UUID, uuid4

import pytest
from hodlin_contracts import canonical_bytes
from hodlin_execute.gate.token import (
    CLOCK_SKEW,
    MIN_SECRET_BYTES,
    TOKEN_SCHEME,
    Binding,
    Rejected,
    Rejection,
    TokenClaims,
    Verified,
    mint,
    verify,
)
from pydantic import ValidationError

_SECRET = b"k" * MIN_SECRET_BYTES
_OTHER_SECRET = b"j" * MIN_SECRET_BYTES

_ISSUED_AT = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
_EXPIRES_AT = _ISSUED_AT + timedelta(minutes=5)
_WITHIN_WINDOW = _ISSUED_AT + timedelta(minutes=1)

_PROPOSAL_HASH = "9dbaa8bdc4719b423311c0a90447a44fb3973ad10a7733941b057f65845a0803"
_OTHER_PROPOSAL_HASH = "94c9ff9d41c074e244a2a9b1bd835b541fc51843ffe75e7f7b201fe66a3d0a68"
_JTI = UUID("3f2a1b0c-4d5e-6f70-8192-a3b4c5d6e7f8")


def _claims(**overrides: Any) -> TokenClaims:
    base: dict[str, Any] = {
        "proposal_hash": _PROPOSAL_HASH,
        "jti": _JTI,
        "subject": "keycloak|operator-42",
        "amount": Decimal("0.25"),
        "recipient_label": "cold-wallet",
        "issued_at": _ISSUED_AT,
        "expires_at": _EXPIRES_AT,
    }
    base.update(overrides)
    return TokenClaims(**base)


def _binding(**overrides: Any) -> Binding:
    """What the gate independently believes it is authorizing."""
    base: dict[str, Any] = {
        "proposal_hash": _PROPOSAL_HASH,
        "amount": Decimal("0.25"),
        "recipient_label": "cold-wallet",
    }
    base.update(overrides)
    return Binding(**base)


def _verify(token: str, **overrides: Any) -> Verified | Rejected:
    kwargs: dict[str, Any] = {
        "secret": _SECRET,
        "expected": _binding(),
        "now": _WITHIN_WINDOW,
    }
    kwargs.update(overrides)
    return verify(token, **kwargs)


def test_a_freshly_minted_token_verifies_and_returns_its_claims() -> None:
    claims = _claims()

    result = _verify(mint(claims, secret=_SECRET))

    assert isinstance(result, Verified)
    assert result.claims == claims
    # The jti comes back because T14 needs it to consume the token exactly once.
    assert result.claims.jti == _JTI


def test_minting_is_deterministic() -> None:
    """Equal claims must produce equal bytes: the MAC covers the serialization, so
    a nondeterministic encoder would make tokens that fail their own verification
    (and would make the canonical-hash reuse pointless)."""
    assert mint(_claims(), secret=_SECRET) == mint(_claims(), secret=_SECRET)


class TestTamper:
    """Any single changed claim must invalidate the token.

    Cases are generated from the model, so a new claim arrives with a test whether
    its author remembers or not. A claim that could be edited without breaking the
    MAC would be a claim the gate cannot trust — which is the same as it not being
    in the token at all.
    """

    #: A different, still-valid value for every field. Keyed by field name so a
    #: newly added claim shows up as a KeyError in the completeness test below,
    #: rather than silently going unexercised.
    ALTERNATIVES: ClassVar[dict[str, Any]] = {
        "proposal_hash": _OTHER_PROPOSAL_HASH,
        "jti": uuid4(),
        "subject": "keycloak|someone-else",
        "amount": Decimal("2.5"),
        "recipient_label": "hot-wallet",
        "issued_at": _ISSUED_AT - timedelta(minutes=1),
        "expires_at": _EXPIRES_AT + timedelta(hours=1),
    }

    def test_every_claim_is_covered_by_a_tamper_case(self) -> None:
        assert set(self.ALTERNATIVES) == set(TokenClaims.model_fields)

    @pytest.mark.parametrize("field", sorted(ALTERNATIVES))
    def test_swapping_the_payload_for_a_tampered_one_fails_the_mac(self, field: str) -> None:
        """The attacker's realistic move: keep the signature, edit the payload.

        Every field is inside the MAC, so this is caught as a signature failure
        before any claim is even parsed — the verifier never has to decide whether
        a *value* is acceptable, which is the property that makes the ordering in
        ``verify`` matter.
        """
        authentic = mint(_claims(), secret=_SECRET)
        _, _, signature = authentic.split(".")
        tampered_payload = canonical_bytes(_claims(**{field: self.ALTERNATIVES[field]}))

        forged = f"{TOKEN_SCHEME}.{_b64u(tampered_payload)}.{signature}"

        assert _verify(forged) == Rejected(Rejection.BAD_SIGNATURE)

    @pytest.mark.parametrize("field", ["proposal_hash", "amount", "recipient_label"])
    def test_a_token_minted_for_other_terms_is_refused_even_though_it_is_authentic(
        self, field: str
    ) -> None:
        """The lifting attack, and the reason the binding exists at all.

        Here the token is genuinely ours — correctly signed, unexpired — but it
        authorizes something other than what the gate is about to do. Without these
        three fields inside the MAC, an approval of 0.25 to the cold wallet would
        be a bearer token for any amount to anywhere.
        """
        token = mint(_claims(**{field: self.ALTERNATIVES[field]}), secret=_SECRET)

        result = _verify(token)

        assert isinstance(result, Rejected)
        assert result.reason in {
            Rejection.PROPOSAL_MISMATCH,
            Rejection.AMOUNT_MISMATCH,
            Rejection.RECIPIENT_MISMATCH,
        }

    def test_the_mismatch_reason_names_the_field(self) -> None:
        """Granular reasons because the audit trail is read by humans deciding what
        happened: "wrong recipient" and "wrong amount" call for different reactions."""
        cold_to_hot = mint(_claims(recipient_label="hot-wallet"), secret=_SECRET)
        bigger = mint(_claims(amount=Decimal("2.5")), secret=_SECRET)
        other_proposal = mint(_claims(proposal_hash=_OTHER_PROPOSAL_HASH), secret=_SECRET)

        assert _verify(cold_to_hot) == Rejected(Rejection.RECIPIENT_MISMATCH)
        assert _verify(bigger) == Rejected(Rejection.AMOUNT_MISMATCH)
        assert _verify(other_proposal) == Rejected(Rejection.PROPOSAL_MISMATCH)


class TestSecret:
    def test_a_token_minted_with_another_secret_is_refused(self) -> None:
        token = mint(_claims(), secret=_OTHER_SECRET)

        assert _verify(token) == Rejected(Rejection.BAD_SIGNATURE)

    def test_a_short_secret_is_refused_loudly_at_both_ends(self) -> None:
        """Not a rejection — a raise. A weak key is our own misconfiguration, not
        untrusted input, and it must not be possible to run with one."""
        with pytest.raises(ValueError, match="at least"):
            mint(_claims(), secret=b"short")
        with pytest.raises(ValueError, match="at least"):
            _verify(mint(_claims(), secret=_SECRET), secret=b"short")


class TestValidityWindow:
    def test_a_token_past_its_expiry_is_refused(self) -> None:
        token = mint(_claims(), secret=_SECRET)

        assert _verify(token, now=_EXPIRES_AT + timedelta(seconds=1)) == Rejected(Rejection.EXPIRED)

    def test_expiry_is_inclusive_of_the_boundary(self) -> None:
        """A token that expires exactly now is spent. On the money path a boundary
        rounds toward refusal: the cost is a human approving again, versus a
        transfer that shouldn't have happened."""
        token = mint(_claims(), secret=_SECRET)

        assert _verify(token, now=_EXPIRES_AT) == Rejected(Rejection.EXPIRED)
        assert isinstance(_verify(token, now=_EXPIRES_AT - timedelta(seconds=1)), Verified)

    def test_a_token_from_the_future_is_refused_beyond_the_skew_tolerance(self) -> None:
        token = mint(_claims(), secret=_SECRET)

        assert _verify(token, now=_ISSUED_AT - CLOCK_SKEW - timedelta(seconds=1)) == Rejected(
            Rejection.NOT_YET_VALID
        )
        assert isinstance(_verify(token, now=_ISSUED_AT - CLOCK_SKEW), Verified)

    def test_a_naive_now_is_a_programming_error(self) -> None:
        """Distinct from a bad token: a caller who forgot the timezone would
        otherwise get a silently wrong comparison on a decision that moves money."""
        token = mint(_claims(), secret=_SECRET)

        with pytest.raises(ValueError, match="timezone-aware"):
            _verify(token, now=datetime(2026, 8, 21, 12, 1))

    def test_claims_with_a_non_positive_window_cannot_be_constructed(self) -> None:
        with pytest.raises(ValidationError):
            _claims(expires_at=_ISSUED_AT)


class TestMalformed:
    """Garbage is an expected input to a verifier, so it returns a value rather
    than raising — a caller can forget a ``try``, but not a return value it has to
    branch on."""

    @pytest.mark.parametrize(
        ("token", "reason"),
        [
            ("", Rejection.MALFORMED),
            ("not-a-token", Rejection.MALFORMED),
            ("a.b", Rejection.MALFORMED),
            ("a.b.c.d", Rejection.MALFORMED),
            ("hodlin-auth-v1.!!!.###", Rejection.MALFORMED),
            ("hodlin-auth-v2.abc.def", Rejection.UNKNOWN_SCHEME),
        ],
        ids=["empty", "no-separators", "too-few-parts", "too-many-parts", "not-base64", "scheme"],
    )
    def test_it_returns_a_reason_instead_of_raising(self, token: str, reason: Rejection) -> None:
        result = _verify(token)

        assert isinstance(result, Rejected)
        assert result.reason is reason

    def test_relabelling_the_scheme_of_a_real_token_fails(self) -> None:
        """The scheme label is inside the MAC (the same reason ``schema_version``
        is inside the proposal hash), so a token cannot be re-presented as a future
        scheme to have different rules applied to it."""
        _, payload, signature = mint(_claims(), secret=_SECRET).split(".")

        assert _verify(f"hodlin-auth-v2.{payload}.{signature}") == Rejected(
            Rejection.UNKNOWN_SCHEME, "hodlin-auth-v2"
        )

    def test_an_authentic_but_nonsensical_payload_is_refused(self) -> None:
        """Signed with our own key, yet not a shape we accept: either our minting
        is broken or the secret leaked. Either way it is not a usable approval, and
        the reason distinguishes it from a forgery."""
        from hodlin_execute.gate.token import _mac

        payload = b'{"proposal_hash":"deadbeef"}'
        signature = _mac(TOKEN_SCHEME, payload, _SECRET)

        result = _verify(f"{TOKEN_SCHEME}.{_b64u(payload)}.{_b64u(signature)}")

        assert isinstance(result, Rejected)
        assert result.reason is Rejection.CLAIMS_INVALID


def test_the_token_is_url_safe_and_carries_no_padding() -> None:
    """It travels in headers and CLI arguments, so ``+``/``/``/``=`` would need
    escaping somewhere and eventually wouldn't be."""
    token = mint(_claims(), secret=_SECRET)

    assert set("+/=").isdisjoint(token)


def test_the_secret_never_appears_in_the_token() -> None:
    """Obvious, and worth pinning: HMAC is not encryption. The payload is readable
    by anyone holding the token — which is fine, since nothing in it is a secret —
    but the key must not be recoverable from it."""
    token = mint(_claims(), secret=_SECRET)

    assert _SECRET.decode() not in token


def _b64u(raw: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
