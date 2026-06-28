"""Shared contracts imported by both domains. The only package both may import.

Frozen ``Proposal`` / ``EvidenceRef`` models plus the canonical-hash helpers.
"""

from hodlin_contracts.canonical import (
    canonical_bytes,
    canonical_hash,
    canonical_json,
)
from hodlin_contracts.proposal import (
    Action,
    EvidenceRef,
    Money,
    Proposal,
)
from hodlin_contracts.version import SCHEMA_VERSION

__all__ = [
    "SCHEMA_VERSION",
    "Action",
    "EvidenceRef",
    "Money",
    "Proposal",
    "canonical_bytes",
    "canonical_hash",
    "canonical_json",
]
