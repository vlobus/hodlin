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
from pydantic import SecretStr, ValidationError

_URL = "postgresql+asyncpg://hodlin_execute:pw@localhost:5432/hodlin_execute"
_SECRET = "s" * 32

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
    for name in ("EXECUTE_DATABASE_URL", "EXECUTE_TOKEN_SECRET", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)


def test_reads_the_prefixed_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXECUTE_DATABASE_URL", _URL)
    monkeypatch.setenv("EXECUTE_TOKEN_SECRET", _SECRET)

    settings = Settings()  # type: ignore[call-arg]  # env-provided

    assert settings.database_url == _URL
    assert settings.token_secret.get_secret_value() == _SECRET


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
    settings = Settings(_env_file=None, database_url=_URL, token_secret=SecretStr(_SECRET))  # type: ignore[call-arg]

    assert settings.database_url == _URL


class TestTokenSecret:
    """The one value that decides whether "a human approved this" means anything."""

    def test_it_is_required(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            Settings(_env_file=None, database_url=_URL)  # type: ignore[call-arg]

        assert "token_secret" in str(excinfo.value)

    def test_a_short_secret_is_refused(self) -> None:
        """A short HMAC key doesn't misbehave — it just makes every token cheap to
        forge, which is the kind of failure that never shows up in a log."""
        with pytest.raises(ValidationError):
            Settings(_env_file=None, database_url=_URL, token_secret=SecretStr("tooshort"))  # type: ignore[call-arg]

    def test_it_does_not_leak_through_repr(self) -> None:
        """Settings objects end up in tracebacks and debug logs; the secret must
        not ride along (D26's no-credential-in-a-log-line rule, applied to the
        value that would be worst to leak)."""
        settings = Settings(_env_file=None, database_url=_URL, token_secret=SecretStr(_SECRET))  # type: ignore[call-arg]

        assert _SECRET not in repr(settings)
        assert _SECRET not in str(settings)
        assert _SECRET not in str(settings.model_dump())

    def test_the_committed_placeholder_is_refused(self) -> None:
        """The finding that mattered most in review: the template's placeholder was
        37 characters, so it passed the length floor and produced a *working*
        system — with an HMAC key published in a public repository.

        Every other placeholder in this repo fails on first use because a remote
        service rejects it. This one has no remote service to refuse it, so the
        refusal has to live here, and a length floor alone can't do it: a
        memorable sentence is easily long enough.
        """
        placeholder = (Path(__file__).resolve().parents[1] / ".env.execute.example").read_text()
        committed = next(
            line.split("=", 1)[1]
            for line in placeholder.splitlines()
            if line.startswith("EXECUTE_TOKEN_SECRET=")
        )

        with pytest.raises(ValidationError):
            Settings(_env_file=None, database_url=_URL, token_secret=SecretStr(committed))  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "value",
        [
            "change-me-at-least-32-characters-long",
            "your-token-secret-goes-right-here-ok",
            "placeholder-placeholder-placeholder",
            "example-secret-for-local-development",
        ],
    )
    def test_other_long_placeholders_are_refused_too(self, value: str) -> None:
        """Not just the exact committed string: the shapes a developer types when
        they mean "I'll do this later" are all long enough to pass a length check."""
        with pytest.raises(ValidationError, match="placeholder"):
            Settings(_env_file=None, database_url=_URL, token_secret=SecretStr(value))  # type: ignore[call-arg]

    def test_a_generated_secret_is_accepted(self) -> None:
        """The check must not be so eager that a real key trips it."""
        generated = "8Kv2mQ7pR4tZ9wX1cB6nH3jL5dF0sA2yE7uI4oP8kM1"

        settings = Settings(_env_file=None, database_url=_URL, token_secret=SecretStr(generated))  # type: ignore[call-arg]

        assert settings.token_secret_bytes == generated.encode("utf-8")

    def test_bytes_conversion_happens_in_one_place(self) -> None:
        """So no call site has to decide an encoding for a key."""
        settings = Settings(_env_file=None, database_url=_URL, token_secret=SecretStr(_SECRET))  # type: ignore[call-arg]

        assert settings.token_secret_bytes == _SECRET.encode("utf-8")


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
