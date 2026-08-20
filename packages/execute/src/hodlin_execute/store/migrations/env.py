"""Alembic environment for the execute domain.

Reads ``EXECUTE_DATABASE_URL`` directly rather than building ``Settings``:
migrations must run without the domain's other secrets (token key, OIDC config).
The prefix means the recommend domain's ``DATABASE_URL`` can't be picked up here
by accident — but it says nothing about *what* the variable points at, so
pointing it at the recommend database would happily create this domain's tables
there. Aim it at the execute database (or a throwaway one, as the tests do).

``version_table`` is named explicitly. The two domains normally live in separate
databases, where the default ``alembic_version`` would be fine — but if someone
ever points both chains at one database, distinct version tables mean they
coexist instead of silently overwriting each other's revision pointer.
"""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from hodlin_execute.store import tables  # noqa: F401  (registers ORM models)
from hodlin_execute.store.db import Base
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import create_async_engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

_configured_url = os.getenv("EXECUTE_DATABASE_URL")
if not _configured_url:
    raise RuntimeError("EXECUTE_DATABASE_URL must be set to run execute-domain migrations")
# Annotated, so the functions below see a ``str`` rather than the ``str | None``
# a module-level narrowing doesn't carry into a nested scope.
database_url: str = _configured_url
# The URL is handed to the engine directly, never through
# ``config.set_main_option``: that stores it in a ConfigParser section, where a
# percent sign is interpolation syntax — a perfectly ordinary percent-encoded
# password (``p%40ss``) would fail the migration with InterpolationSyntaxError
# instead of connecting.

target_metadata = Base.metadata

VERSION_TABLE = "alembic_version_execute"


def _run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_async_migrations() -> None:
    connectable = create_async_engine(database_url, poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(_run_migrations)
    await connectable.dispose()


def run_migrations_offline() -> None:
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=VERSION_TABLE,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(_run_async_migrations())
