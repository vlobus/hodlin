"""Shared contracts imported by both domains. The only package both may import.

Frozen proposal/evidence models plus the canonical-hash helpers. Proposal
versions live side by side (D32): ``ProposalV1_0`` is what M1 shipped and must
keep hashing identically forever, ``ProposalV1_1`` is what this codebase now
produces (``SCHEMA_VERSION``), and ``parse_proposal`` reads either by dispatching
on the version carried in the payload.
"""

from hodlin_contracts.canonical import (
    canonical_bytes,
    canonical_hash,
    canonical_json,
)
from hodlin_contracts.proposal import (
    ActionV1_0,
    ActionV1_1,
    AnyProposal,
    EvidenceRef,
    Money,
    ProposalLike,
    ProposalV1_0,
    ProposalV1_1,
    UtcDatetime,
    is_expired,
    parse_proposal,
)
from hodlin_contracts.version import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_1_0,
    SCHEMA_VERSION_1_1,
)

__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_1_0",
    "SCHEMA_VERSION_1_1",
    "ActionV1_0",
    "ActionV1_1",
    "AnyProposal",
    "EvidenceRef",
    "Money",
    "ProposalLike",
    "ProposalV1_0",
    "ProposalV1_1",
    "UtcDatetime",
    "canonical_bytes",
    "canonical_hash",
    "canonical_json",
    "is_expired",
    "parse_proposal",
]
