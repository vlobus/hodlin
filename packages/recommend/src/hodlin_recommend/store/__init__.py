"""Persistence layer: async engine/session, ORM tables, repositories.

This is the only part of the recommend domain that talks SQL. Domain logic
depends on the repositories, never on SQLAlchemy directly (repository pattern).
"""
