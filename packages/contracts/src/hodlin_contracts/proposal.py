"""Frozen, validated contracts exchanged between the two domains (D7, D14).

A proposal is what the recommend domain produces and the execute domain
consumes. Every version is immutable (``frozen=True``), rejects unknown fields
(``extra="forbid"``), carries money as ``Decimal`` so it stays exact (never a
float), uses timezone-aware datetimes only, and requires at least one piece of
evidence. It deliberately carries **no raw destination address** — the
recipient is named by label and resolved to an address inside the execute
domain at tx-build time, so a prompt-injected recommend domain can't direct
funds anywhere (D14).

**Two versions live here side by side** (D32, T12): ``ProposalV1_0`` exactly as
M1 shipped it — byte-for-byte, so its historical digests still reproduce — and
``ProposalV1_1``, which adds the ``transfer`` action and an optional
``valid_until``. There is deliberately no bare ``Proposal`` alias: a name whose
meaning silently moves to the newest version is precisely the ambiguity that
versioning exists to remove. Call sites that mean "whatever we produce today"
use ``SCHEMA_VERSION``; call sites that accept input use ``parse_proposal``,
which dispatches on the version *in the payload*.

Freshness is not enforced here. ``valid_until`` in the past parses fine, because
a historical proposal must stay parseable forever — re-reading a stored proposal
is not the same act as accepting a new one. Expiry is a decision for the gate
(T16), which asks ``is_expired`` at the moment of approval.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Literal, Protocol, Self, cast, runtime_checkable
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    model_validator,
)

from hodlin_contracts.version import SCHEMA_VERSION_1_0, SCHEMA_VERSION_1_1


def _reject_float(value: object) -> object:
    """Money must never originate from a float — that reintroduces the
    binary-rounding error the Decimal type exists to avoid. A string or
    Decimal parses exactly; a float does not."""
    if isinstance(value, float):
        raise ValueError("money must be a Decimal or string, not a float")
    return value


def _normalize_money(value: Decimal) -> Decimal:
    """Reduce a money amount to a single canonical form so meaning-equal
    amounts hash identically: ``0.50`` and ``0.5`` must serialize the same.
    ``normalize`` strips trailing zeros but can yield exponent notation for
    whole numbers (``Decimal("1E+2")``); rebuilding from the fixed-point
    string form keeps the canonical value plain (never scientific) without
    ``quantize``, which would raise ``InvalidOperation`` on amounts whose
    digit count exceeds the decimal context precision. ``Decimal`` parses a
    string exactly, free of context limits."""
    normalized = value.normalize()
    if normalized == 0:
        return Decimal(0)  # collapse signed/scaled zeros (e.g. -0, 0E+2) to "0"
    return Decimal(format(normalized, "f"))


def _to_utc(value: datetime) -> datetime:
    """Pin every timestamp to UTC so the same instant has one representation:
    ``12:05Z`` and ``13:05+01:00`` are equal in meaning and must hash alike."""
    return value.astimezone(UTC)


Money = Annotated[Decimal, BeforeValidator(_reject_float), AfterValidator(_normalize_money)]

UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]

#: 1.0's actions. Frozen with the version — every value here is inside a digest
#: that has to keep reproducing.
ActionV1_0 = Literal["buy", "sell", "hold", "alert"]

#: 1.1 adds ``transfer``: the action that actually moves value, and therefore the
#: reason the execute gate exists. Adding it to 1.0 would have been the mutation
#: D32 forbids — a 1.0 proposal's digest must not depend on what 1.1 allows.
ActionV1_1 = Literal["buy", "sell", "hold", "alert", "transfer"]


class _Frozen(BaseModel):
    """Base config shared by every contract: immutable, no unknown fields,
    surrounding whitespace stripped (so a ``min_length=1`` field can't be
    satisfied by spaces, and canonical strings stay stable)."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class EvidenceRef(_Frozen):
    """A single citable source behind a proposal — a news item, a price
    anomaly, or a sentiment score. At least one is required on every
    proposal so a recommendation can always be traced back to what it saw.

    **Shared across proposal versions, and therefore inside every version's
    freeze.** That is a real asymmetry with ``ActionV1_0``/``ActionV1_1``, so it
    needs saying: widening ``kind`` here would widen what a *frozen 1.0* document
    is allowed to say, without bumping 1.0. Existing digests would still
    reproduce, which is exactly what makes it easy to miss. It stays shared
    because the shape is genuinely identical across versions and duplicating it
    would invite the two copies to drift — but the allowed ``kind`` values are
    pinned by a test, so adding one is a deliberate act that forces the
    "does this need a new proposal version?" conversation instead of sliding by.
    """

    kind: Literal["anomaly", "news", "sentiment", "price"]
    source: str = Field(min_length=1)
    ref: str = Field(min_length=1)
    observed_at: UtcDatetime


class ProposalV1_0(_Frozen):
    """An AI-authored recommendation, schema 1.0 — **exactly as M1 shipped it**.

    Nothing serialized here may change, ever. Its digests are already computed
    and (from slice C on) minted into tokens, so a single added or renamed field
    would silently invalidate every stored hash. New shapes go in a new class;
    ``tests/fixtures/proposal_v1_0.json`` pins this one's digest as a regression.
    """

    schema_version: Literal["1.0"] = SCHEMA_VERSION_1_0
    proposal_id: UUID
    asset: str = Field(min_length=1)
    action: ActionV1_0
    amount: Money = Field(ge=0)
    recipient_label: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    created_at: UtcDatetime


class ProposalV1_1(_Frozen):
    """Schema 1.1 — 1.0 plus the ``transfer`` action and an optional
    ``valid_until``. Purely additive in meaning, and still a *separate document*:
    the same business fields hash differently under 1.1 because
    ``schema_version`` is inside the hash (D32).

    ``valid_until`` bounds how long the *proposal* is worth acting on, which is
    not the same clock as the approval token's ``expires_at`` (T13): the first
    says "this recommendation is stale", the second says "this authorization is
    spent". A proposal can be fresh with an expired token and vice versa, so both
    exist.
    """

    schema_version: Literal["1.1"] = SCHEMA_VERSION_1_1
    proposal_id: UUID
    asset: str = Field(min_length=1)
    action: ActionV1_1
    amount: Money = Field(ge=0)
    recipient_label: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    created_at: UtcDatetime
    #: Optional deadline. ``None`` means the proposal states no expiry of its own
    #: — the gate still bounds it by the approval token's lifetime.
    valid_until: UtcDatetime | None = None

    @model_validator(mode="after")
    def _value_moving_actions_need_an_amount(self) -> Self:
        """``ge=0`` is right for ``hold``/``alert``, which carry no amount — but a
        zero-amount ``transfer`` (or buy, or sell) is not something any human would
        approve: it burns gas to do nothing, and it's the shape a prompt-injected
        recommend domain emits when it's flailing. 1.1 is where the value-moving
        action arrives, so it's where the floor belongs. 1.0 keeps its old rules
        untouched — tightening validation on a frozen version could make a stored
        proposal unparseable, which is the mutation D32 forbids in another guise.
        """
        if self.action in ("buy", "sell", "transfer") and self.amount == 0:
            raise ValueError(f"{self.action} requires an amount greater than zero")
        return self

    @model_validator(mode="after")
    def _valid_until_after_created_at(self) -> Self:
        """A deadline at or before the moment of creation is self-contradictory —
        the proposal was never actionable. Rejecting it here is safe precisely
        because the check compares two of the proposal's *own* fields: the verdict
        never changes with the passage of time, so no stored proposal can become
        unparseable later. Anything that depends on "now" belongs at the gate.
        """
        if self.valid_until is not None and self.valid_until <= self.created_at:
            raise ValueError("valid_until must be after created_at")
        return self


#: Any proposal version this codebase can read, tagged by the version in the
#: payload. A discriminated union rather than a try-each-model cascade: dispatch
#: is explicit, and an unknown version fails as "no such tag" instead of as a
#: pile of confusing field errors from every candidate.
AnyProposal = Annotated[ProposalV1_0 | ProposalV1_1, Field(discriminator="schema_version")]

_PROPOSAL_ADAPTER: TypeAdapter[ProposalV1_0 | ProposalV1_1] = TypeAdapter(AnyProposal)


def parse_proposal(payload: Mapping[str, object]) -> ProposalV1_0 | ProposalV1_1:
    """Parse a proposal of *any* known version, dispatching on ``schema_version``.

    Note the deliberate asymmetry with direct construction: ``ProposalV1_1(...)``
    defaults the version, because in-process we know what we're building — but a
    payload arriving from outside must **say** which contract it is. Guessing on
    behalf of a caller is how a 1.0 document gets read as a 1.1 one, and the
    digest that guess produces would be wrong in a way nothing downstream can
    detect. Raises ``pydantic.ValidationError`` for a missing or unknown version.
    """
    return _PROPOSAL_ADAPTER.validate_python(payload)


@runtime_checkable
class ProposalLike(Protocol):
    """Structural view of "some version of a proposal".

    Exists so version-spanning helpers don't have to enumerate classes. An
    enumeration is the wrong shape for a module whose whole thesis is that new
    shapes arrive as new classes: the compiler is perfectly happy when a widened
    union reaches an ``isinstance`` chain that silently stops matching, and on
    this path "no match" would mean *not expired* — failing open, on the money
    side, for a version that plainly stated a deadline.
    """

    @property
    def schema_version(self) -> str: ...


def is_expired(proposal: ProposalLike, at: datetime) -> bool:
    """Whether the proposal states a deadline that ``at`` has passed.

    Freshness is asked as a question, not enforced at parse time (see the module
    docstring). 1.0 has no ``valid_until`` and so never expires *of its own
    accord* — which is not a loophole: the gate's other bounds (a short-lived,
    single-use token) still apply, and they're what make a 1.0 approval
    non-replayable.

    The deadline is read *structurally*, not by matching known classes, so a
    future version that carries ``valid_until`` is honoured the moment it exists
    rather than the moment someone remembers to extend this function. Opting in
    by accident is fine here; opting out by accident is a stale proposal that the
    gate believes it checked.
    """
    if at.tzinfo is None:
        raise ValueError("`at` must be timezone-aware to compare against valid_until")
    deadline = cast("datetime | None", getattr(proposal, "valid_until", None))
    return deadline is not None and at > deadline
