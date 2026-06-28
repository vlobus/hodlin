"""Validate the Alembic migration applies and reverses against real Postgres.

This is the production schema path ("Alembic owns the schema"); the idempotency
test builds its schema from the ORM metadata, so this independently proves the
hand-written migration matches and is reversible.
"""

from pathlib import Path

import pytest

_RECOMMEND_DIR = Path(__file__).resolve().parents[2] / "packages" / "recommend"
_ALEMBIC_INI = _RECOMMEND_DIR / "alembic.ini"
_SCRIPT_LOCATION = _RECOMMEND_DIR / "src" / "hodlin_recommend" / "store" / "migrations"


def test_migration_upgrade_then_downgrade(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alembic import command
    from alembic.config import Config

    monkeypatch.setenv("DATABASE_URL", postgres_url)
    config = Config(str(_ALEMBIC_INI))
    config.set_main_option("script_location", str(_SCRIPT_LOCATION))

    command.upgrade(config, "head")
    command.downgrade(config, "base")
