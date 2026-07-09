"""Async engine + session machinery (SQLAlchemy 2.0 + asyncpg).

``Base`` is the declarative root every ORM table inherits from; its
``metadata`` is what Alembic diffs against. The engine is a connection pool to
Postgres — created once at startup and shared. Sessions are short-lived units
of work handed to repositories.
"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models; carries the shared ``metadata``."""


# The type components ask for when they mint their own sessions (jobs, the
# readiness probe) — an alias so their signatures don't spell out generics.
SessionFactory = async_sessionmaker[AsyncSession]


def create_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    """Build the async engine (a pooled connection to Postgres). ``pool_pre_ping``
    quietly recycles connections the DB dropped, so a restarted Postgres doesn't
    surface as a stale-connection error on the next query."""
    return create_async_engine(database_url, echo=echo, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """A factory that mints sessions bound to the engine. ``expire_on_commit=False``
    keeps loaded objects usable after commit (we read attributes post-commit)."""
    return async_sessionmaker(engine, expire_on_commit=False)
