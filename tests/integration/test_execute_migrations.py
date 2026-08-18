"""The execute domain's Alembic chain applies and reverses against real Postgres.

Same shape as the recommend domain's migration test, and for the same reason:
Alembic owns the schema, so the hand-written migration must be provably
equivalent to ``store/tables.py`` and provably reversible.

It also demonstrates the coexistence property from D34: this chain stamps
``alembic_version_execute``, so pointing both domains' chains at one database
(which these tests do) leaves each domain's revision pointer intact instead of
one silently clobbering the other.

Note the ``to_thread`` calls: the migration env drives the async engine with
``asyncio.run``, which refuses to nest inside a running loop, so an async test
has to hand the command to a thread that has no loop of its own.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from hodlin_execute.store.db import Base as ExecuteBase
from hodlin_execute.store.db import create_engine
from sqlalchemy import Connection, inspect
from sqlalchemy.ext.asyncio import AsyncEngine

_EXECUTE_DIR = Path(__file__).resolve().parents[2] / "packages" / "execute"
_ALEMBIC_INI = _EXECUTE_DIR / "alembic.ini"
_SCRIPT_LOCATION = _EXECUTE_DIR / "src" / "hodlin_execute" / "store" / "migrations"

_EXPECTED_TABLES = {
    "operators",
    "roles",
    "permissions",
    "role_permissions",
    "operator_roles",
    "proposals",
    "auth_tokens",
    "approvals",
    "tx_attempts",
}


def _only_execute_tables(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    """Restrict the comparison to this domain's tables. These tests point both
    domains' chains at one database, so everything else in it — the recommend
    tables, both ``alembic_version*`` pointers — is out of scope by definition."""
    if type_ == "table":
        return name in ExecuteBase.metadata.tables
    return True


def _config(postgres_url: str, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("EXECUTE_DATABASE_URL", postgres_url)
    config = Config(str(_ALEMBIC_INI))
    config.set_main_option("script_location", str(_SCRIPT_LOCATION))
    return config


def test_migration_upgrade_then_downgrade(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(postgres_url, monkeypatch)

    command.upgrade(config, "head")
    command.downgrade(config, "base")


async def test_migration_creates_every_orm_table(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The migration and the ORM metadata must not drift apart: everything
    ``tables.py`` declares has to exist after an upgrade, including the partial
    unique index that enforces one live token per proposal."""
    config = _config(postgres_url, monkeypatch)
    await asyncio.to_thread(command.upgrade, config, "head")

    engine: AsyncEngine = create_engine(postgres_url)
    try:
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda sync: set(inspect(sync).get_table_names()))
            indexes = await conn.run_sync(
                lambda sync: {ix["name"] for ix in inspect(sync).get_indexes("auth_tokens")}
            )
        assert _EXPECTED_TABLES <= tables
        assert set(ExecuteBase.metadata.tables) <= tables
        assert "uq_auth_tokens_one_live_per_proposal" in indexes
    finally:
        await engine.dispose()
        await asyncio.to_thread(command.downgrade, config, "base")


async def test_migration_does_not_drift_from_the_orm_metadata(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The strong form of the check above: not "every table exists" but "the
    database Alembic built is *indistinguishable* from ``tables.py``".

    Table names surviving an upgrade is a weak signal — a column renamed on one
    side only, or an int32 where the ORM says int64, passes it happily. Alembic's
    own autogenerate comparison is what would produce the next revision, so an
    empty diff is the statement that the hand-written migration and the ORM say
    the same thing.
    """
    config = _config(postgres_url, monkeypatch)
    await asyncio.to_thread(command.upgrade, config, "head")

    engine: AsyncEngine = create_engine(postgres_url)
    try:

        def diff(sync: Connection) -> list[Any]:
            context = MigrationContext.configure(
                sync, opts={"include_object": _only_execute_tables}
            )
            return list(compare_metadata(context, ExecuteBase.metadata))

        async with engine.connect() as conn:
            differences = await conn.run_sync(diff)
        assert differences == []
    finally:
        await engine.dispose()
        await asyncio.to_thread(command.downgrade, config, "base")
