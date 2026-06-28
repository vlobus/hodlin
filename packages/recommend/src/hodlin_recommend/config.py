"""Runtime configuration for the recommend domain.

Loaded once at the composition root (``main.py``). Only the store needs
settings in T3; connector keys, Telegram, and Anthropic are added by their
own tasks. Values come from the environment (or a local ``.env``), so secrets
never live in code.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Async SQLAlchemy URL — note the ``+asyncpg`` driver. Overridden per
    # environment (docker-compose, tests) via the DATABASE_URL env var.
    database_url: str = "postgresql+asyncpg://hodlin:hodlin@localhost:5432/hodlin"
