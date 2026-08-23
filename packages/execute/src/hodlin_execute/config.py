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

The OIDC settings and chain settings arrive with their own tasks (T15, T17).
"""

from typing import Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from hodlin_execute.gate.token import MIN_SECRET_BYTES

#: Substrings that mean "nobody generated this yet". A length floor alone can't
#: catch a placeholder — a memorable sentence is easily long enough — and a
#: placeholder HMAC key is the one fake credential that *works*, silently, with a
#: value anyone can read in the repository. Every other secret here fails on
#: first use because the remote service rejects it; this one has no remote
#: service to refuse it, so the refusal has to live here.
_PLACEHOLDER_MARKERS = ("generate", "change-me", "changeme", "your-", "example", "placeholder")


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

    # The HMAC key every authorization token is minted and verified with (T13,
    # D15). ``SecretStr`` so it can't be printed by an accidental ``repr`` of the
    # settings object — the value that makes every token forgeable is exactly the
    # one that must never reach a log line. Length is enforced here as well as in
    # the token module, because a short key is a silent weakening rather than a
    # failure: nothing misbehaves, the MAC is just cheap to brute-force.
    token_secret: SecretStr = Field(min_length=MIN_SECRET_BYTES)

    @model_validator(mode="after")
    def _token_secret_is_not_a_placeholder(self) -> Self:
        secret = self.token_secret.get_secret_value().casefold()
        for marker in _PLACEHOLDER_MARKERS:
            if marker in secret:
                raise ValueError(
                    f"token_secret looks like the committed placeholder (contains {marker!r}); "
                    "generate one with `openssl rand -base64 48`"
                )
        return self

    @property
    def token_secret_bytes(self) -> bytes:
        """The key as the HMAC wants it. One place does this conversion, so no
        call site has to decide an encoding for a key."""
        return self.token_secret.get_secret_value().encode("utf-8")
