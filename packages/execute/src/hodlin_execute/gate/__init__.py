"""The approval gate: what turns a human decision into permission to move money.

``token`` is pure (HMAC mint/verify, no I/O). ``replay_store`` (T14) adds the
atomic single-use consume, and ``approval`` (T16) composes authorize → mint →
audit into one unit of work. Expiry and single-use are deliberately separate
mechanisms in separate modules, because they are separate properties (D15).
"""

from hodlin_execute.gate.replay_store import (
    Consumed,
    ConsumedReason,
    ConsumeRejected,
    ConsumeRejection,
    ConsumeResult,
    Registered,
    RegisterRejected,
    RegisterRejection,
    RegisterResult,
    ReplayStore,
)
from hodlin_execute.gate.token import (
    CLOCK_SKEW,
    MAX_TOKEN_TTL,
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
    "MAX_TOKEN_TTL",
    "MIN_SECRET_BYTES",
    "TOKEN_SCHEME",
    "Binding",
    "ConsumeRejected",
    "ConsumeRejection",
    "ConsumeResult",
    "Consumed",
    "ConsumedReason",
    "RegisterRejected",
    "RegisterRejection",
    "RegisterResult",
    "Registered",
    "Rejected",
    "Rejection",
    "ReplayStore",
    "TokenClaims",
    "Verified",
    "VerifyResult",
    "mint",
    "verify",
]
