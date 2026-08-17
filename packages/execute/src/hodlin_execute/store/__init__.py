"""Persistence layer for the execute domain: engine/session, ORM tables, repositories.

The only part of this domain that talks SQL, against its own database with its
own credential (D34). Gate and chain logic depend on repositories, never on
SQLAlchemy directly.
"""
