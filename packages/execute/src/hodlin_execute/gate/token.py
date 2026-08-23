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
import hmac
from dataclasses import dataclass
from datetime import datetime, timedelta
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

#: The longest window this module will mint or accept. Everything here and in the
#: replay store is justified by the token being *short-lived*, and until now
#: nothing enforced that — ``timedelta(days=5)`` where ``minutes=5`` was meant
#: would have produced a five-day bearer token that verified perfectly. Checked
#: at both ends: at mint because it's our own bug, and at verify because an
#: authentic token with an absurd window is exactly what a compromised minter
#: emits.
MAX_TOKEN_TTL = timedelta(minutes=15)

#: Tolerance for a token that appears to have been minted a moment in the future.
#: One process mints and verifies within the same domain, so this is jitter, not
#: distributed clock reconciliation — deliberately small, because a wide window
#: is indistinguishable from not checking. Applied only to ``issued_at``: adding
#: it to the expiry check would *extend* validity, which is the one direction a
#: tolerance must never move on the money path.
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
    #: Bounded to match ``operators.subject`` — a claim that mints fine and then
    #: fails at INSERT would fail *after* the human approved, on the write that
    #: records the decision.
    subject: str = Field(min_length=1, max_length=255)
    #: Bound so a token can't be lifted onto a bigger transfer. ``gt=0``, not
    #: ``ge=0``: a token authorizes *moving value*, and a zero-value move isn't
    #: something to authorize (T12 rejects it in 1.1 proposals for the same
    #: reason). Note the precondition that creates for T16 — an approved ``hold``
    #: or ``alert`` proposal is legally zero-amount, and minting a token for one
    #: would raise here. Those actions have nothing to execute, so the gate must
    #: refuse them as a *domain* outcome before reaching this module; an exception
    #: on the approval path is the wrong shape for "there is nothing to authorize".
    amount: Money = Field(gt=0)
    #: Bound so a token can't be lifted onto a different destination. The *label*,
    #: not an address — the address is resolved execute-side from config (D14).
    #: Bounded to match ``tx_attempts.recipient_label``, for the same reason as
    #: ``subject``.
    recipient_label: str = Field(min_length=1, max_length=64)
    issued_at: UtcDatetime
    expires_at: UtcDatetime

    @model_validator(mode="after")
    def _window_is_positive(self) -> Self:
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be after issued_at")
        return self


class Binding(BaseModel):
    """What the caller believes it is authorizing, checked against the claims.

    The gate recomputes these from its own state — the stored proposal, its own
    hash of it — and hands them here. Verification then answers "is this token
    for *this* act", not merely "is this token authentic".

    Validated as strictly as the claims it is compared against, which is
    correctness rather than symmetry for its own sake: an upper-case digest, a
    ``float`` amount (``0.1 != Decimal("0.1")``), or an unstripped
    ``" cold-wallet "`` would each produce a perfectly plausible ``*_MISMATCH``.
    A *caller's* bug would then be recorded in the audit trail as "the token was
    for a different recipient" — a lie that costs someone an afternoon.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    proposal_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    amount: Money = Field(gt=0)
    recipient_label: str = Field(min_length=1, max_length=64)


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
    WINDOW_TOO_LONG = "window_too_long"


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


def _safe_detail(text: str, *, limit: int = 32) -> str:
    """Make attacker-controlled text safe to put in a log line or an audit row.

    ``Rejection`` details reach ``approvals.reason`` (a ``Text`` column) and log
    lines, so echoing a caller's bytes verbatim hands them a channel: a megabyte
    of newlines and ANSI escapes, delivered by one bogus token. Printable ASCII
    only, hard length cap.
    """
    printable = "".join(ch for ch in text if ch.isascii() and ch.isprintable())
    return printable[:limit]


def _secret_bytes(secret: bytes) -> bytes:
    if len(secret) < MIN_SECRET_BYTES:
        raise ValueError(
            f"token secret must be at least {MIN_SECRET_BYTES} bytes, got {len(secret)}"
        )
    return secret


def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64u_decode(text: str) -> bytes:
    """Decode base64url, accepting **only** the canonical spelling of the bytes.

    ``validate=True`` alone is not enough, and the reason is worth spelling out
    because it's easy to believe otherwise. ``b64decode`` translates ``altchars``
    *before* applying its validation regex, so ``+`` and ``/`` still pass; and
    validation says nothing about non-canonical trailing bits, so several distinct
    final characters decode to identical bytes. The effect is that one authentic
    token has many equally-valid spellings.

    Nothing today breaks because of that — T14 dedupes on ``jti``, not on the
    token string — but "the token string is canonical" is exactly the kind of
    assumption a later replay cache, idempotency key, or log-based duplicate
    detector would be built on, and it would be bypassable by changing one
    character. So the decode is checked by re-encoding: if the round trip isn't
    identical, the token is malformed.
    """
    padding = "=" * (-len(text) % 4)
    raw = base64.b64decode(text + padding, altchars=b"-_", validate=True)
    if _b64u_encode(raw) != text:
        raise ValueError("non-canonical base64url encoding")
    return raw


def _mac(scheme: str, payload: bytes, secret: bytes) -> bytes:
    """HMAC over the scheme label *and* the payload, framed unambiguously.

    The label is length-prefixed rather than separated by a delimiter. A
    delimiter argument would have been wrong here — the canonical payload
    contains ``.`` (an amount serializes as ``"0.25"``), so a separator-based
    framing relies on the label being checked against a constant beforehand,
    which is a different guarantee than the framing itself providing it. Length
    prefixing means no (scheme, payload) pair can produce the same MAC input as
    any other, which is what keeps this correct when a key id joins the label.
    """
    label = scheme.encode("ascii")
    framed = len(label).to_bytes(2, "big") + label + payload
    return hmac.new(secret, framed, sha256).digest()


def mint(claims: TokenClaims, *, secret: bytes) -> str:
    """Serialize and MAC the claims into a token string.

    The payload is the *canonical* JSON of the claims (the same serialization the
    proposal hash uses), so a token minted twice from equal claims is byte-equal —
    which matters because the MAC is over those exact bytes.
    """
    if claims.expires_at - claims.issued_at > MAX_TOKEN_TTL:
        raise ValueError(
            f"token window {claims.expires_at - claims.issued_at} exceeds "
            f"MAX_TOKEN_TTL ({MAX_TOKEN_TTL})"
        )
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
        return Rejected(Rejection.UNKNOWN_SCHEME, _safe_detail(scheme))

    try:
        payload = _b64u_decode(payload_b64)
        signature = _b64u_decode(signature_b64)
    except ValueError:
        # One clause: ``binascii.Error`` subclasses ``ValueError``, which also
        # catches the non-canonical-encoding check above.
        return Rejected(Rejection.MALFORMED, "payload or signature is not canonical base64url")

    # compare_digest, never ==: an early-exit comparison leaks the correct MAC
    # byte by byte through timing, and the attacker here gets to retry.
    if not hmac.compare_digest(signature, _mac(scheme, payload, secret)):
        return Rejected(Rejection.BAD_SIGNATURE)

    try:
        claims = TokenClaims.model_validate_json(payload)
    except ValidationError as exc:
        # Authentic but nonsensical: our own minting is broken, or the secret
        # leaked and someone is minting shapes we don't accept.
        # The detail names the offending *fields*, never their values: an
        # operator needs to know which claim was unacceptable, and the values are
        # attacker-supplied.
        fields = sorted({str(error["loc"][0]) for error in exc.errors() if error["loc"]})
        # A longer cap than the echo paths get: these names come from our own
        # model, not from the caller, and a list truncated at 32 characters hides
        # exactly the field the operator needed to see.
        return Rejected(Rejection.CLAIMS_INVALID, _safe_detail(",".join(fields), limit=200))

    if claims.proposal_hash != expected.proposal_hash:
        return Rejected(Rejection.PROPOSAL_MISMATCH)
    if claims.amount != expected.amount:
        return Rejected(Rejection.AMOUNT_MISMATCH)
    if claims.recipient_label != expected.recipient_label:
        return Rejected(Rejection.RECIPIENT_MISMATCH)

    if claims.expires_at - claims.issued_at > MAX_TOKEN_TTL:
        # Authentic, and still not a token this system will honour: either our
        # minting regressed or someone holding the secret is issuing long-lived
        # bearer tokens. Refusing here makes "short-lived" a property of the gate
        # rather than a property of whoever called mint.
        return Rejected(Rejection.WINDOW_TOO_LONG)
    if now + CLOCK_SKEW < claims.issued_at:
        return Rejected(Rejection.NOT_YET_VALID)
    # ``>=``: a token that expires exactly now is spent. Boundaries on the money
    # path round toward refusal, since the cost of one extra approval round-trip
    # is a human clicking again, and the cost of the other direction is a transfer.
    if now >= claims.expires_at:
        return Rejected(Rejection.EXPIRED)

    return Verified(claims)
