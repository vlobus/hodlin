"""Async engine + session machinery for the execute domain's own database.

Deliberately a near-duplicate of the recommend domain's ``store/db.py`` rather
than a shared helper: the only package both domains may import is
``hodlin_contracts`` (D7/D16), which exists to carry the *messages* that cross
the boundary — not infrastructure. Coupling the two domains' persistence layers
through it to save twenty lines would trade a real isolation property for a
trivial DRY win. Each domain also owns a **separate** ``Base``, so its Alembic
chain diffs only its own tables (D34).
"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for the execute domain's ORM models."""


SessionFactory = async_sessionmaker[AsyncSession]


def create_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    """Build the async engine. ``pool_pre_ping`` recycles connections the
    server dropped, so a restarted Postgres doesn't surface as a stale
    connection on the next query."""
    return create_async_engine(database_url, echo=echo, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """A factory minting sessions bound to the engine. ``expire_on_commit=False``
    keeps loaded objects readable after commit."""
    return async_sessionmaker(engine, expire_on_commit=False)
