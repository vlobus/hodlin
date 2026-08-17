"""Alembic environment for the execute domain.

Reads ``EXECUTE_DATABASE_URL`` directly rather than building ``Settings``:
migrations must run without the domain's other secrets (token key, OIDC config),
and the prefix keeps it impossible to accidentally migrate the *recommend*
database from here (D34).

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
from sqlalchemy.ext.asyncio import async_engine_from_config

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

database_url = os.getenv("EXECUTE_DATABASE_URL")
if not database_url:
    raise RuntimeError("EXECUTE_DATABASE_URL must be set to run execute-domain migrations")
config.set_main_option("sqlalchemy.url", database_url)

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
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_run_migrations)
    await connectable.dispose()


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
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
