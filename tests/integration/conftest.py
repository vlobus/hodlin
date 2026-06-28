"""Fixtures for store integration tests — a real Postgres via testcontainers.

The container is started once per session; each test gets a fresh schema
(created from the ORM metadata and dropped afterwards) and its own session.
If Docker or the image isn't available the fixtures ``skip`` rather than fail,
so the suite stays green on machines without Docker (CI has it).
"""

from collections.abc import AsyncIterator, Iterator

import pytest
from hodlin_recommend.store.db import Base, create_engine, create_session_factory
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:  # pragma: no cover
        pytest.skip("testcontainers not installed")
    try:
        with PostgresContainer("postgres:16-alpine", driver="asyncpg") as postgres:
            yield postgres.get_connection_url()
    except Exception as exc:  # pragma: no cover - docker absent / image pull failed
        pytest.skip(f"postgres testcontainer unavailable: {exc}")


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
