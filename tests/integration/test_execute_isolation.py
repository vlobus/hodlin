"""The two domains cannot reach each other's data (T11, D34).

import-linter proves the *code* can't cross the boundary. This proves the
*credentials* can't: each domain has its own database owned by its own
non-superuser role, and PUBLIC has no CONNECT on either. So a fully compromised
recommend domain — prompt injection, SQL injection, a stolen container — still
cannot open a connection to the database holding authorization tokens,
approvals, and RBAC grants. Postgres has no cross-database queries, so being
unable to connect ends the conversation.

The provisioning SQL under test is the very same file docker-compose mounts into
the postgres image, so this asserts the deployed configuration rather than a
restatement of it.
"""

from pathlib import Path

import asyncpg
import pytest
from sqlalchemy.engine import make_url

_INIT_SQL = Path(__file__).resolve().parents[2] / "docker" / "postgres-init" / "10-domains.sql"

_RECOMMEND = ("hodlin_recommend", "hodlin_recommend", "hodlin_recommend")
_EXECUTE = ("hodlin_execute", "hodlin_execute", "hodlin_execute")


def _split_statements(sql: str) -> list[str]:
    """Split the init file into individual statements.

    Naive ``sql.split(";")`` would cut the ``DO $$ ... $$`` block in half, and
    the statements can't simply be sent as one multi-statement query either:
    Postgres wraps those in an implicit transaction, and ``CREATE DATABASE``
    refuses to run inside one. So: track dollar-quoting, emit one statement at a
    time.
    """
    statements: list[str] = []
    buffer: list[str] = []
    in_dollar_quote = False
    for line in sql.splitlines():
        stripped = line.strip()
        if not in_dollar_quote and (not stripped or stripped.startswith("--")):
            continue
        if line.count("$$") % 2 == 1:
            in_dollar_quote = not in_dollar_quote
        buffer.append(line)
        if not in_dollar_quote and stripped.endswith(";"):
            statements.append("\n".join(buffer))
            buffer = []
    return statements


async def _connect(
    url_template: str, user: str, password: str, database: str
) -> asyncpg.Connection:
    url = make_url(url_template)
    return await asyncpg.connect(
        user=user,
        password=password,
        database=database,
        host=url.host,
        port=url.port,
    )


@pytest.fixture
async def provisioned(postgres_url: str) -> str:
    """Apply the committed init SQL, skipping if this Postgres won't allow it."""
    url = make_url(postgres_url)
    admin = await asyncpg.connect(
        user=url.username,
        password=url.password,
        database=url.database,
        host=url.host,
        port=url.port,
    )
    try:
        is_superuser = await admin.fetchval(
            "SELECT rolsuper FROM pg_roles WHERE rolname = current_user"
        )
        if not is_superuser:
            pytest.skip(
                "provisioning roles/databases needs a superuser; point "
                "HODLIN_TEST_DATABASE_URL at an admin account or let "
                "testcontainers provide one"
            )
        for statement in _split_statements(_INIT_SQL.read_text()):
            try:
                await admin.execute(statement)
            except asyncpg.DuplicateObjectError, asyncpg.DuplicateDatabaseError:
                # Already provisioned (a dev compose db, or a re-run) — the
                # file is written to be safe to reapply.
                pass
    finally:
        await admin.close()
    return postgres_url


async def test_each_domain_can_reach_its_own_database(provisioned: str) -> None:
    """Sanity first: the isolation must be selective, not a wall around
    everything — each role has to be able to work in its own database."""
    for user, password, database in (_RECOMMEND, _EXECUTE):
        conn = await _connect(provisioned, user, password, database)
        try:
            assert await conn.fetchval("SELECT current_database()") == database
            # The role owns its database, so it must be able to create its own
            # schema — otherwise Alembic couldn't migrate.
            await conn.execute("CREATE TABLE IF NOT EXISTS _isolation_probe (id int)")
            await conn.execute("DROP TABLE _isolation_probe")
        finally:
            await conn.close()


async def test_recommend_credential_cannot_connect_to_the_execute_database(
    provisioned: str,
) -> None:
    """The property that matters: owning the recommend domain buys you nothing
    in the execute domain."""
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        await _connect(provisioned, _RECOMMEND[0], _RECOMMEND[1], _EXECUTE[2])


async def test_execute_credential_cannot_connect_to_the_recommend_database(
    provisioned: str,
) -> None:
    """And symmetrically — the authority-holding domain has no business reading
    market data, so it can't."""
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        await _connect(provisioned, _EXECUTE[0], _EXECUTE[1], _RECOMMEND[2])


async def test_public_has_no_connect_privilege_on_either_database(provisioned: str) -> None:
    """Postgres grants CONNECT to PUBLIC by default, which would quietly undo
    the whole arrangement for any future role — assert the revoke stuck."""
    url = make_url(provisioned)
    admin = await asyncpg.connect(
        user=url.username,
        password=url.password,
        database=url.database,
        host=url.host,
        port=url.port,
    )
    try:
        for database in (_RECOMMEND[2], _EXECUTE[2]):
            granted = await admin.fetchval(
                "SELECT has_database_privilege('public', $1, 'CONNECT')", database
            )
            assert granted is False, f"PUBLIC still has CONNECT on {database}"
    finally:
        await admin.close()
