"""Runtime configuration for the execute domain.

Every value comes from the environment with **no in-code default** for anything
secret (D17), so a missing value fails loudly at startup rather than silently
running with something weak.

Two mechanisms, doing two different jobs (D34), because the prefix alone is
often mistaken for the whole boundary:

* the ``EXECUTE_`` prefix means this domain cannot *accidentally bind* the
  recommend domain's bare ``DATABASE_URL`` — leak that variable into this
  process and it configures nothing (asserted in tests/test_execute_config.py);
* the separate env file (``.env.execute``, never ``.env``) is what keeps the two
  domains' secrets from being *distributed together*. A prefix cannot protect a
  credential that is sitting in a file the other domain's container is handed;
  only the split can, since compose injects every key of an ``env_file``.

So: two domains sharing a Postgres server share no credential, and neither
process is ever handed the other's.

The token secret, OIDC settings, and chain settings arrive with their own tasks
(T13, T15, T17).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Execute-domain settings, read from ``EXECUTE_*`` env vars (or ``.env.execute``)."""

    # ``.env.execute``, not ``.env``: this domain's values are distributed
    # separately from the recommend domain's (see the module docstring).
    model_config = SettingsConfigDict(
        env_file=".env.execute", env_prefix="EXECUTE_", extra="ignore"
    )

    # Async SQLAlchemy URL for the execute domain's *own* database — note the
    # ``+asyncpg`` driver, and that this points at ``hodlin_execute``, reachable
    # only by the ``hodlin_execute`` role (docker/postgres-init/10-domains.sql).
    database_url: str
