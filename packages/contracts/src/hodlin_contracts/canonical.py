"""Canonical serialization + content hash for contracts.

One implementation, imported by both domains. The rules (JCS-flavoured): keys
sorted, no insignificant whitespace, nulls explicit, Decimals as strings. The
SHA-256 over those bytes is a stable content identity — in slice C it's what
the execute-domain HMAC token is minted over, so two proposals that are equal
in meaning must hash identically regardless of how their fields happened to be
ordered at construction.
"""

import hashlib
import json

from pydantic import BaseModel


def canonical_json(model: BaseModel) -> str:
    """Deterministic JSON for any contract model: sorted keys, no whitespace,
    Decimals/datetimes already stringified by ``mode="json"``."""
    payload = model.model_dump(mode="json")
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(model: BaseModel) -> bytes:
    return canonical_json(model).encode("utf-8")


def canonical_hash(model: BaseModel) -> str:
    """Hex-encoded SHA-256 of the canonical bytes."""
    return hashlib.sha256(canonical_bytes(model)).hexdigest()
