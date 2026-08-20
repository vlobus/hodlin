"""Execute domain: holds authority, moves money.

The half of the system that may act. It accepts a proposal it did not author,
requires a verified human's approval, and only then builds and broadcasts a
transaction — so authority never comes from text an agent processed (D7, D14).

Its isolation is enforced three ways, not one: import-linter forbids any
recommend ↔ execute import (D7/D16), it holds its own secrets under an
``EXECUTE_`` prefix, and it owns a separate database reachable only by its own
Postgres role (D34).

T11 lands the store layer (RBAC, approvals, the single-use token replay store,
on-chain attempt records). The token itself (T13), its atomic consume (T14),
OIDC identity + RBAC checks (T15), the gate API (T16), and the Sepolia
transaction path (T17) follow.
"""
