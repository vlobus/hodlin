"""Runtime configuration for the recommend domain.

Loaded once at the composition root (``main.py``). Values come from the
environment (or a local ``.env``), so secrets never live in code. Telegram and
Anthropic settings are added by their own tasks.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration is externalized to the environment (twelve-factor): every
    field is required and read from an env var (or ``.env``), with no in-code
    default, so a missing/typo'd value fails loudly at startup. See
    ``.env.example`` for the full set. Components receive these via DI, so tests
    pass explicit values and never depend on a global ``Settings()``."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Async SQLAlchemy URL — note the ``+asyncpg`` driver.
    database_url: str

    # Finnhub — company news (D4).
    finnhub_api_key: str
    finnhub_base_url: str
    finnhub_rate_per_min: float

    # Massive — price OHLC bars (D13).
    massive_api_key: str
    massive_base_url: str
    massive_rate_per_min: float
