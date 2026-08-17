"""Runtime configuration for the execute domain.

Every value comes from the environment with **no in-code default** for anything
secret (D17), so a missing value fails loudly at startup rather than silently
running with something weak.

The ``EXECUTE_`` prefix is a security boundary, not a naming convention (D34):
compose hands this service only its own ``EXECUTE_*`` variables, so the
recommend domain's connection string and API keys are absent from this
process's environment entirely — and vice versa. Two domains that share a
Postgres server share no credential.

The token secret, OIDC settings, and chain settings arrive with their own tasks
(T13, T15, T17).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Execute-domain settings, read from ``EXECUTE_*`` env vars (or ``.env``)."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="EXECUTE_", extra="ignore")

    # Async SQLAlchemy URL for the execute domain's *own* database — note the
    # ``+asyncpg`` driver, and that this points at ``hodlin_execute``, reachable
    # only by the ``hodlin_execute`` role (docker/postgres-init/10-domains.sql).
    database_url: str
