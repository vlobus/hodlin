"""Persistence layer for the execute domain: engine/session, ORM tables, repositories.

Almost the only part of this domain that talks SQL, against its own database with
its own credential (D34). Gate and chain logic depend on repositories rather than
on SQLAlchemy directly — with one deliberate exception, worth naming because an
unexplained exception to a stated rule is worse than either following it or
dropping it.

``gate/replay_store.py`` writes its own statements. Its guarantee *is* a specific
statement — one conditional ``UPDATE … WHERE consumed_at IS NULL`` — and hiding
that behind a generic repository method would put the mechanism somewhere a reader
of the gate can't see it, while adding an indirection whose only content is the
statement it obscures. The single-use property is the point of that module, so the
SQL that provides it lives there, next to the reasoning.
"""
