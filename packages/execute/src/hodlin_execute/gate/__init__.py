"""The approval gate: what turns a human decision into permission to move money.

``token`` is pure — HMAC mint/verify, no clock, no database. ``replay_store``
(T14) adds the atomic single-use consume, and ``approval`` (T16) will compose
authorize → mint → audit into one unit of work. Expiry and single-use live in
separate modules because they are separate properties (D15).

**Deliberately no re-exports.** An eager ``from .replay_store import …`` here
would mean importing the pure token module pulls in SQLAlchemy and the whole ORM
metadata — quietly undoing the property ``token.py``'s docstring claims for
itself, and making "pure" a comment rather than a fact you can check with an
import. Import the module you actually need.
"""
