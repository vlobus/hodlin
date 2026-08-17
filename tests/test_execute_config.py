"""Execute-domain settings: fail-fast, and prefixed so secrets can't cross over.

The ``EXECUTE_`` prefix is the point of these tests rather than a naming detail
(D34): if this domain also answered to the recommend domain's bare
``DATABASE_URL``, then handing one container the other's environment would
silently work, and the isolation would be a comment rather than a mechanism.
"""

import pytest
from hodlin_execute.config import Settings
from pydantic import ValidationError

_URL = "postgresql+asyncpg://hodlin_execute:pw@localhost:5432/hodlin_execute"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # A developer's shell (or compose env) must not decide these assertions;
    # _env_file=None below closes the dotenv path, this closes os.environ.
    for name in ("EXECUTE_DATABASE_URL", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)


def test_reads_the_prefixed_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXECUTE_DATABASE_URL", _URL)

    assert Settings().database_url == _URL  # type: ignore[call-arg]  # env-provided


def test_ignores_the_recommend_domains_unprefixed_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leaking the recommend domain's DATABASE_URL into this process must not
    configure the execute domain — it must still fail for want of its own."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://hodlin_recommend:pw@localhost/x")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_missing_database_url_fails_loudly() -> None:
    """No in-code default (D17): a missing value is a startup error, not a
    quietly-wrong connection."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)  # type: ignore[call-arg]

    assert "database_url" in str(excinfo.value)


def test_explicit_values_bypass_the_environment_for_tests() -> None:
    settings = Settings(_env_file=None, database_url=_URL)  # type: ignore[call-arg]

    assert settings.database_url == _URL
