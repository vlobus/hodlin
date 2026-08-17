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
-- and by tests/integration/test_execute_isolation.py, which asserts the
-- properties below actually hold. NOTE the image only runs init scripts when
-- the data directory is empty — an existing volume needs one
-- `docker compose down -v` to pick this up.
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

-- OWNED BY matters beyond tidiness: in PG15+ the `public` schema belongs to
-- pg_database_owner, so making each role its own database's owner is what
-- gives it CREATE rights for its Alembic migrations — without granting it
-- anything anywhere else.
CREATE DATABASE hodlin_recommend OWNER hodlin_recommend;
CREATE DATABASE hodlin_execute OWNER hodlin_execute;

-- Default Postgres grants CONNECT on every database to PUBLIC. Revoke it and
-- hand it back to exactly one role each; this is the line that makes the two
-- domains unable to reach each other's data.
REVOKE CONNECT ON DATABASE hodlin_recommend FROM PUBLIC;
REVOKE CONNECT ON DATABASE hodlin_execute FROM PUBLIC;
GRANT CONNECT ON DATABASE hodlin_recommend TO hodlin_recommend;
GRANT CONNECT ON DATABASE hodlin_execute TO hodlin_execute;
