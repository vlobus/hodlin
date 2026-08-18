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
restatement of it. Two consequences of taking that literally:

* the file is applied **twice**, because its idempotence is itself a security
  property — a re-application that aborts before the REVOKE would leave PUBLIC
  with its default CONNECT, and nothing else here would notice;
* the statement splitter below has to read the file the way psql does, ``\\gexec``
  and inline comments included, rather than the way that happens to work today.
"""

from dataclasses import dataclass
from pathlib import Path

import asyncpg
import pytest
from sqlalchemy.engine import make_url

_INIT_SQL = Path(__file__).resolve().parents[2] / "docker" / "postgres-init" / "10-domains.sql"

_RECOMMEND = ("hodlin_recommend", "hodlin_recommend", "hodlin_recommend")
_EXECUTE = ("hodlin_execute", "hodlin_execute", "hodlin_execute")

_GEXEC = "\\gexec"


@dataclass(frozen=True)
class _Statement:
    """One statement from the init file. ``gexec`` marks psql's ``\\gexec``: run
    the statement, then run every row it returns as a statement of its own."""

    sql: str
    gexec: bool


def _split_statements(sql: str) -> list[_Statement]:
    """Split the init file into individual statements, the way psql reads it.

    The statements can't be sent as one multi-statement query: Postgres wraps
    those in an implicit transaction, and ``CREATE DATABASE`` refuses to run
    inside one. So they're split here — and the split has to survive dollar
    quoting (``DO $$ ... $$`` contains semicolons), string literals, and inline
    ``--`` comments (a trailing comment after a semicolon must not swallow the
    statement it follows, which is exactly how a silently-skipped REVOKE would
    happen). Anything left unterminated at EOF raises rather than vanishing.
    """
    statements: list[_Statement] = []
    buffer: list[str] = []
    in_string = False
    in_dollar_quote = False
    index = 0

    def flush(gexec: bool) -> None:
        statement = "".join(buffer).strip()
        buffer.clear()
        if statement:
            statements.append(_Statement(statement, gexec))

    while index < len(sql):
        if sql.startswith("$$", index):
            if not in_string:
                in_dollar_quote = not in_dollar_quote
            buffer.append("$$")
            index += 2
            continue
        char = sql[index]
        if in_dollar_quote:
            buffer.append(char)
            index += 1
            continue
        if in_string:
            if char == "'":
                in_string = False
            buffer.append(char)
            index += 1
            continue
        if char == "'":
            in_string = True
        elif sql.startswith("--", index):
            newline = sql.find("\n", index)
            index = len(sql) if newline == -1 else newline
            continue
        elif char == ";":
            buffer.append(char)
            flush(gexec=False)
            index += 1
            continue
        elif sql.startswith(_GEXEC, index):
            flush(gexec=True)
            index += len(_GEXEC)
            continue
        buffer.append(char)
        index += 1

    leftover = "".join(buffer).strip()
    assert not leftover, f"unterminated statement in {_INIT_SQL.name}: {leftover!r}"
    return statements


async def _apply(admin: asyncpg.Connection, sql: str) -> None:
    for statement in _split_statements(sql):
        if statement.gexec:
            for row in await admin.fetch(statement.sql):
                await admin.execute(row[0])
        else:
            await admin.execute(statement.sql)


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
    """Apply the committed init SQL twice, skipping if this Postgres won't allow it.

    Twice, and with no ``except`` around it: reapplying must be a clean no-op,
    because the deployed script runs under ``ON_ERROR_STOP=1`` where a single
    duplicate-object error would abort the file before the REVOKE that carries
    the isolation. Swallowing the error here would make this suite pass on a
    script that fails *open* in production.
    """
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
        sql = _INIT_SQL.read_text()
        await _apply(admin, sql)
        await _apply(admin, sql)
    finally:
        await admin.close()
    return postgres_url


def test_the_init_file_splits_into_the_statements_psql_would_run() -> None:
    """The splitter is load-bearing — it decides which of the file's statements
    this suite actually applies — so its two failure modes are pinned here: a
    dollar-quoted body must stay one statement, and a trailing comment must not
    swallow the statement it follows."""
    statements = _split_statements(
        "DO $$ BEGIN PERFORM 1; PERFORM 2; END $$;\n"
        "SELECT 'x' WHERE false\n"
        "\\gexec\n"
        "REVOKE CONNECT ON DATABASE d FROM PUBLIC;  -- the line that matters\n"
    )

    assert [(s.sql.split()[0], s.gexec) for s in statements] == [
        ("DO", False),
        ("SELECT", True),
        ("REVOKE", False),
    ]
    assert statements[0].sql.count("PERFORM") == 2  # the DO block stayed whole
    assert statements[2].sql.endswith("PUBLIC;")  # comment stripped, statement kept


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
    the whole arrangement for any future role — assert the revoke stuck. Because
    the fixture applied the file twice, this also asserts that a *reapplication*
    still lands the revoke instead of aborting on the existing databases."""
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
