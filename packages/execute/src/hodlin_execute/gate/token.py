"""The authorization token: HMAC-SHA256 mint and verify, pure (T13, D15).

An approval by a human becomes a *token* — the only thing that lets the chain leg
run. This module is the whole of that mechanism and deliberately holds nothing
else: no clock, no database, no config lookup. Time arrives as an argument and the
secret arrives as an argument, which is what makes every property below testable
without a fixture and impossible to get accidentally right in production but wrong
under test.

**Why HMAC and not a signature.** The same trust boundary mints and verifies: the
execute domain issues the token to itself, minutes later. A symmetric secret is
the correct tool for that shape, and it sidesteps the algorithm-confusion family
of bugs that asymmetric JWTs invite (``alg: none``, HS-signed-with-the-public-key).
The moment minting moves to a *separate* service, this becomes the wrong choice
and an asymmetric signature becomes right — that's the upgrade D15 records.

**What the token binds, and why each field is in there.** A MAC over the proposal
hash alone would be liftable: attach it to a different amount or a different
recipient and the digest still checks out. So the claims carry the amount and the
recipient label *as authorized*, plus the approver's subject (attribution), a
``jti`` (the handle T14's single-use store consumes), and the validity window.
Anything the gate will later rely on has to be inside the MAC, because everything
outside it is attacker-controlled.

**Expiry is not single-use.** Everything here is stateless, so it can only bound
*how long* an approval is good for. "Exactly once" is a different property needing
different machinery — an atomic compare-and-set on ``jti`` in Postgres (T14). Two
properties, two mechanisms; neither substitutes for the other.
"""

import base64
import binascii
import hmac
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import Self
from uuid import UUID

from hodlin_contracts import Money, UtcDatetime, canonical_bytes
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

#: Scheme label, carried in the token *and* inside the MAC. Relabelling a token
#: to a future scheme therefore invalidates it, rather than getting it
#: reinterpreted under different rules — the same reason ``schema_version`` sits
#: inside the proposal hash (D32). A key id for secret rotation would live
#: alongside this label, and would have to be covered by the MAC for the same
#: reason; rotation itself is out of scope for M2.
TOKEN_SCHEME = "hodlin-auth-v1"

#: The minimum secret this module will accept. A short HMAC key is the one
#: configuration mistake that silently weakens everything else here, so it is a
#: hard error rather than a warning.
MIN_SECRET_BYTES = 32

#: Tolerance for a token that appears to have been minted a moment in the future.
#: One process mints and verifies within the same domain, so this is jitter, not
#: distributed clock reconciliation — deliberately small, because a wide window
#: is indistinguishable from not checking.
CLOCK_SKEW = timedelta(seconds=5)


class TokenClaims(BaseModel):
    """Everything the token asserts. Frozen, and every field is inside the MAC.

    Adding a field here changes what an approval means, so it also changes what
    must be verified — ``tests/test_gate_token.py`` derives its tamper cases from
    this model's fields, which makes "new claim, no test" impossible rather than
    merely discouraged.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    #: Canonical hash of the proposal as *this domain* computed it (never as a
    #: caller reported it) — the anchor the whole approval hangs off.
    proposal_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    #: Single-use handle. Stateless verification can't enforce that; T14's
    #: compare-and-set on this value can.
    jti: UUID
    #: OIDC ``sub`` of the human who approved. Attribution belongs inside the MAC:
    #: an audit trail that can be edited by the caller is not an audit trail.
    subject: str = Field(min_length=1)
    #: Bound so a token can't be lifted onto a bigger transfer.
    amount: Money = Field(gt=0)
    #: Bound so a token can't be lifted onto a different destination. The *label*,
    #: not an address — the address is resolved execute-side from config (D14).
    recipient_label: str = Field(min_length=1)
    issued_at: UtcDatetime
    expires_at: UtcDatetime

    @model_validator(mode="after")
    def _window_is_positive(self) -> Self:
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be after issued_at")
        return self


@dataclass(frozen=True, slots=True)
class Binding:
    """What the caller believes it is authorizing, checked against the claims.

    The gate recomputes these from its own state — the stored proposal, its own
    hash of it — and hands them here. Verification then answers "is this token
    for *this* act", not merely "is this token authentic".
    """

    proposal_hash: str
    amount: Decimal
    recipient_label: str


class Rejection(StrEnum):
    """Why a token was refused, at the granularity the audit trail needs.

    One generic failure would be easier to write and useless to operate: "the
    token was tampered with" and "the token was for a different recipient" call
    for different human responses, and T16 records the reason on the approval row.
    """

    MALFORMED = "malformed"
    UNKNOWN_SCHEME = "unknown_scheme"
    BAD_SIGNATURE = "bad_signature"
    CLAIMS_INVALID = "claims_invalid"
    PROPOSAL_MISMATCH = "proposal_mismatch"
    AMOUNT_MISMATCH = "amount_mismatch"
    RECIPIENT_MISMATCH = "recipient_mismatch"
    NOT_YET_VALID = "not_yet_valid"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class Verified:
    """The token is authentic, for this act, and inside its window."""

    claims: TokenClaims


@dataclass(frozen=True, slots=True)
class Rejected:
    """The token is not usable, and why. Note there is no exception path for
    attacker-controlled input: garbage is an *expected outcome* of a verifier, so
    it returns a value the caller must handle rather than an exception the caller
    might forget to catch."""

    reason: Rejection
    detail: str = ""


type VerifyResult = Verified | Rejected


def _secret_bytes(secret: bytes) -> bytes:
    if len(secret) < MIN_SECRET_BYTES:
        raise ValueError(
            f"token secret must be at least {MIN_SECRET_BYTES} bytes, got {len(secret)}"
        )
    return secret


def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64u_decode(text: str) -> bytes:
    """Strict base64url decode.

    ``validate=True`` matters more than it looks: by default Python *discards*
    characters outside the alphabet, so ``"ab!!cd"`` and ``"abcd"`` decode to the
    same bytes and obvious garbage decodes to something rather than failing. That
    would make many distinct token strings share one payload — a needless
    ambiguity in the one string this system treats as authority, and it would
    report a malformed token as a signature failure, which reads very differently
    in an audit log.
    """
    padding = "=" * (-len(text) % 4)
    # ``b64decode`` with explicit altchars rather than ``urlsafe_b64decode``,
    # which takes no ``validate`` flag — and strictness is the whole point here.
    # It also means the standard ``+``/``/`` alphabet is refused, so exactly one
    # spelling of a token decodes.
    return base64.b64decode(text + padding, altchars=b"-_", validate=True)


def _mac(scheme: str, payload: bytes, secret: bytes) -> bytes:
    """HMAC over the scheme label *and* the payload.

    Concatenating with a separator that cannot occur in either part (the label is
    ASCII without ``.``, the payload is raw JSON bytes) keeps the two fields from
    being confusable — otherwise a crafted label could borrow bytes from the
    payload and produce the same MAC input as a different pair.
    """
    return hmac.new(secret, f"{scheme}.".encode("ascii") + payload, sha256).digest()


def mint(claims: TokenClaims, *, secret: bytes) -> str:
    """Serialize and MAC the claims into a token string.

    The payload is the *canonical* JSON of the claims (the same serialization the
    proposal hash uses), so a token minted twice from equal claims is byte-equal —
    which matters because the MAC is over those exact bytes.
    """
    payload = canonical_bytes(claims)
    signature = _mac(TOKEN_SCHEME, payload, _secret_bytes(secret))
    return f"{TOKEN_SCHEME}.{_b64u_encode(payload)}.{_b64u_encode(signature)}"


def verify(token: str, *, secret: bytes, expected: Binding, now: datetime) -> VerifyResult:
    """Check a token's authenticity, then its binding, then its validity window.

    That order is the point. Authenticity comes first because every later check
    reads values from the payload, and reading *unauthenticated* values to decide
    anything is how a verifier gets talked out of its own conclusion. Only after
    the MAC holds do the claims get parsed at all.

    ``now`` is a required argument, not a call to the clock, so "expired" is a
    property this module can be asked about rather than a behaviour that depends
    on when the test suite runs. A naive ``now`` is a programming error and raises;
    a malformed *token* is an expected input and returns ``Rejected``.
    """
    if now.tzinfo is None:
        raise ValueError("`now` must be timezone-aware")
    secret = _secret_bytes(secret)

    parts = token.split(".")
    if len(parts) != 3:
        return Rejected(Rejection.MALFORMED, "expected scheme.payload.signature")
    scheme, payload_b64, signature_b64 = parts

    # Compared before the MAC only to produce a useful reason; the label is inside
    # the MAC too, so a relabelled token fails there regardless.
    if scheme != TOKEN_SCHEME:
        return Rejected(Rejection.UNKNOWN_SCHEME, scheme)

    try:
        payload = _b64u_decode(payload_b64)
        signature = _b64u_decode(signature_b64)
    except binascii.Error, ValueError:
        return Rejected(Rejection.MALFORMED, "payload or signature is not base64url")

    # compare_digest, never ==: an early-exit comparison leaks the correct MAC
    # byte by byte through timing, and the attacker here gets to retry.
    if not hmac.compare_digest(signature, _mac(scheme, payload, secret)):
        return Rejected(Rejection.BAD_SIGNATURE)

    try:
        claims = TokenClaims.model_validate_json(payload)
    except ValidationError as exc:
        # Authentic but nonsensical: our own minting is broken, or the secret
        # leaked and someone is minting shapes we don't accept.
        return Rejected(Rejection.CLAIMS_INVALID, str(exc.error_count()))

    if claims.proposal_hash != expected.proposal_hash:
        return Rejected(Rejection.PROPOSAL_MISMATCH)
    if claims.amount != expected.amount:
        return Rejected(Rejection.AMOUNT_MISMATCH)
    if claims.recipient_label != expected.recipient_label:
        return Rejected(Rejection.RECIPIENT_MISMATCH)

    if now + CLOCK_SKEW < claims.issued_at:
        return Rejected(Rejection.NOT_YET_VALID)
    # ``>=``: a token that expires exactly now is spent. Boundaries on the money
    # path round toward refusal, since the cost of one extra approval round-trip
    # is a human clicking again, and the cost of the other direction is a transfer.
    if now >= claims.expires_at:
        return Rejected(Rejection.EXPIRED)

    return Verified(claims)
