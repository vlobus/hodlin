"""Fixtures for store integration tests — a real Postgres.

Two ways to provide the database, in priority order:

1. ``HODLIN_TEST_DATABASE_URL`` — point the suite at a Postgres you already run
   (``docker compose up -d db``, a native install, anything). No Docker Hub pull;
   a bad URL fails loudly, since you explicitly opted in.
2. Otherwise an ephemeral Postgres via testcontainers. If Docker can't provide
   one (daemon down, or the image can't be pulled), the tests ``skip`` so the
   suite stays green where Docker is unavailable (e.g. CI without it).

Each test gets a fresh schema (created from the ORM metadata, dropped after) and
its own session. The URL must use the async driver: ``postgresql+asyncpg://``.
"""

import os
from collections.abc import AsyncIterator, Iterator

import pytest
from hodlin_recommend.store.db import Base, create_engine, create_session_factory
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    explicit = os.getenv("HODLIN_TEST_DATABASE_URL")
    if explicit:
        yield explicit
        return
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:  # pragma: no cover
        pytest.skip("testcontainers not installed")
    try:
        with PostgresContainer("postgres:18-alpine", driver="asyncpg") as postgres:
            yield postgres.get_connection_url()
    except Exception as exc:  # pragma: no cover - docker absent / image pull failed
        pytest.skip(
            "no test Postgres: set HODLIN_TEST_DATABASE_URL to an existing "
            f"postgresql+asyncpg:// database, or enable Docker ({exc})"
        )


@pytest.fixture
async def engine(postgres_url: str) -> AsyncIterator[AsyncEngine]:
    eng = create_engine(postgres_url)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    factory = create_session_factory(engine)
    async with factory() as sess:
        yield sess
