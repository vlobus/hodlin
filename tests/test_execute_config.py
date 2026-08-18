"""Execute-domain settings: fail-fast, prefixed, and distributed separately.

Two mechanisms, and these tests keep them apart (D34), because conflating them
is how a boundary turns into a comment:

* the ``EXECUTE_`` prefix — if this domain also answered to the recommend
  domain's bare ``DATABASE_URL``, handing one container the other's environment
  would silently work;
* the separate env file — a prefix cannot protect a credential that sits in a
  file the other domain's container is handed wholesale (compose injects every
  key of an ``env_file``), so the two domains' values must not share one file.
"""

from pathlib import Path

import pytest
from hodlin_execute.config import Settings
from pydantic import ValidationError

_URL = "postgresql+asyncpg://hodlin_execute:pw@localhost:5432/hodlin_execute"

_ROOT = Path(__file__).resolve().parents[1]
_RECOMMEND_ENV_EXAMPLE = _ROOT / ".env.example"
_EXECUTE_ENV_EXAMPLE = _ROOT / ".env.execute.example"


def _assignments(env_file: Path) -> list[str]:
    """The variable names an env file assigns, comments and blanks aside."""
    return [
        line.split("=", 1)[0].strip()
        for line in env_file.read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    ]


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


def test_the_execute_domains_settings_read_their_own_env_file() -> None:
    """Not ``.env``: that file is the recommend domain's, and compose hands it to
    the recommend container in full."""
    assert Settings.model_config["env_file"] == ".env.execute"


def test_the_two_domains_env_templates_share_no_variables() -> None:
    """The templates are what a developer copies, and what compose then mounts
    per service — so co-locating the two domains' variables in one of them is the
    leak, whatever the code does afterwards. The recommend file must carry no
    ``EXECUTE_*``, and the execute file must carry nothing else."""
    recommend = _assignments(_RECOMMEND_ENV_EXAMPLE)
    execute = _assignments(_EXECUTE_ENV_EXAMPLE)

    assert execute, "the execute template must list the variables this domain reads"
    assert [name for name in execute if name.startswith("EXECUTE_")] == execute
    assert [name for name in recommend if name.startswith("EXECUTE_")] == []
    assert set(recommend).isdisjoint(execute)
