-- Two domains, two databases, two logins (T11, D34).
--
-- The import boundary (D7) stops the code from crossing; this stops the
-- *credentials* from crossing. Each domain gets its own database owned by its
-- own non-superuser role, and PUBLIC loses CONNECT on both — so a fully
-- compromised recommend domain (prompt injection, SQL injection, stolen
-- container) still cannot open a connection to the execute database where the
-- authorization tokens, approvals, and RBAC grants live. Postgres has no
-- cross-database queries, so "can't connect" is the whole story.
--
-- Run automatically by the postgres image from /docker-entrypoint-initdb.d,
-- and by tests/integration/test_execute_isolation.py, which applies this file
-- TWICE and then asserts the properties below actually hold. NOTE the image
-- only runs init scripts when the data directory is empty — an existing volume
-- needs one `docker compose down -v` to pick this up.
--
-- Every statement here is idempotent, and that is a security property rather
-- than a convenience: psql runs init scripts under ON_ERROR_STOP=1, so an
-- unguarded `CREATE DATABASE` against a cluster that already has one (a re-run,
-- or databases pre-created by IaC) would abort the script *before* the REVOKE
-- below — leaving PUBLIC holding its default CONNECT. That failure mode opens
-- the boundary instead of closing it, so it has to be impossible, not unlikely.
--
-- The passwords here are local-development values, deliberately visible, the
-- same as compose's existing hodlin/hodlin. A real deployment provisions these
-- roles from IaC with a secrets manager and never from a committed file.

-- Roles are created conditionally so re-running this against a database that
-- already has them (a dev Postgres, a second test run) is a no-op.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hodlin_recommend') THEN
        CREATE ROLE hodlin_recommend LOGIN PASSWORD 'hodlin_recommend';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hodlin_execute') THEN
        CREATE ROLE hodlin_execute LOGIN PASSWORD 'hodlin_execute';
    END IF;
END
$$;

-- CREATE DATABASE has no IF NOT EXISTS and cannot run inside a DO block (it
-- refuses to run in a transaction at all), so the statement is *generated* only
-- for the databases that are missing and executed by psql's \gexec. Zero rows
-- means zero statements: applying this file again does nothing here and falls
-- through to the grants below.
--
-- OWNED BY matters beyond tidiness: in PG15+ the `public` schema belongs to
-- pg_database_owner, so making each role its own database's owner is what
-- gives it CREATE rights for its Alembic migrations — without granting it
-- anything anywhere else.
SELECT format('CREATE DATABASE %I OWNER %I', d.name, d.name)
  FROM (VALUES ('hodlin_recommend'), ('hodlin_execute')) AS d(name)
 WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = d.name)
\gexec

-- Default Postgres grants CONNECT on every database to PUBLIC. Revoke it and
-- hand it back to exactly one role each; this is the line that makes the two
-- domains unable to reach each other's data. Both statements are naturally
-- idempotent, and they run on every application of this file — including the
-- ones where the databases already existed.
REVOKE CONNECT ON DATABASE hodlin_recommend FROM PUBLIC;
REVOKE CONNECT ON DATABASE hodlin_execute FROM PUBLIC;
GRANT CONNECT ON DATABASE hodlin_recommend TO hodlin_recommend;
GRANT CONNECT ON DATABASE hodlin_execute TO hodlin_execute;
