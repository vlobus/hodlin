"""Alembic environment — runs migrations against the async engine.

The database URL comes from the DATABASE_URL env var (set by docker-compose,
CI, or tests) and falls back to the app default, so no connection string is
baked into ``alembic.ini``. ``target_metadata`` is the ORM's metadata, which
``--autogenerate`` diffs to produce new revisions.
"""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from hodlin_recommend.config import Settings
from hodlin_recommend.store import tables  # noqa: F401  (registers ORM models)
from hodlin_recommend.store.db import Base
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", os.getenv("DATABASE_URL", Settings().database_url))

target_metadata = Base.metadata


def _run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
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
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(_run_async_migrations())
