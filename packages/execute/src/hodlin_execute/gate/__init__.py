"""The approval gate: what turns a human decision into permission to move money.

``token`` is pure (HMAC mint/verify, no I/O). ``replay_store`` (T14) adds the
atomic single-use consume, and ``approval`` (T16) composes authorize → mint →
audit into one unit of work. Expiry and single-use are deliberately separate
mechanisms in separate modules, because they are separate properties (D15).
"""

from hodlin_execute.gate.token import (
    CLOCK_SKEW,
    MIN_SECRET_BYTES,
    TOKEN_SCHEME,
    Binding,
    Rejected,
    Rejection,
    TokenClaims,
    Verified,
    VerifyResult,
    mint,
    verify,
)

__all__ = [
    "CLOCK_SKEW",
    "MIN_SECRET_BYTES",
    "TOKEN_SCHEME",
    "Binding",
    "Rejected",
    "Rejection",
    "TokenClaims",
    "Verified",
    "VerifyResult",
    "mint",
    "verify",
]
