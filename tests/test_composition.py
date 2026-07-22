"""Composition-root tests — the provider selection that demo mode flips.

The full ``build_components`` constructs FinBERT (heavy, ~440 MB) so it isn't
exercised here; what's testable in isolation and actually branches is the
source selection, so that's what we pin. Everything else in the composition is
straight-line wiring verified by the app booting.
"""

import httpx
import pytest
from hodlin_recommend.composition import select_bar_source, select_news_source
from hodlin_recommend.config import Settings
from hodlin_recommend.connectors.base import NewsSource, PriceBarSource
from hodlin_recommend.connectors.finnhub import FinnhubNewsSource
from hodlin_recommend.connectors.massive import MassivePriceBarSource
from hodlin_recommend.connectors.offline import NullNewsSource
from hodlin_recommend.connectors.seed_bars import SeedBarSource

_BASE = {
    "database_url": "postgresql+asyncpg://x:x@localhost/x",
    "finnhub_api_key": "fk",
    "finnhub_base_url": "https://finnhub.test",
    "finnhub_rate_per_min": 60,
    "massive_api_key": "mk",
    "massive_base_url": "https://massive.test",
    "massive_rate_per_min": 5,
    "anthropic_api_key": "ak",
    "anthropic_model": "claude-x",
    "telegram_bot_token": "tk",
    "telegram_base_url": "https://api.telegram.test",
    "telegram_chat_id": 42,
    "telegram_rate_per_min": 30,
}


def _settings(**overrides: object) -> Settings:
    # _env_file=None so a developer's local .env can't leak into these
    # assertions (host/port/demo_mode aren't in _BASE, so they'd otherwise
    # come from .env if present).
    # _env_file is a pydantic-settings runtime init kwarg mypy doesn't model.
    return Settings(_env_file=None, **{**_BASE, **overrides})  # type: ignore[arg-type, call-arg]


@pytest.fixture
async def client() -> httpx.AsyncClient:
    async with httpx.AsyncClient() as c:
        return c


def test_defaults_are_non_secret_and_production_leaning() -> None:
    settings = _settings()
    assert settings.host == "127.0.0.1"  # loopback unless a deploy overrides it
    assert settings.port == 8000
    assert settings.demo_mode is False  # live providers by default


async def test_live_mode_selects_the_real_providers(client: httpx.AsyncClient) -> None:
    settings = _settings(demo_mode=False)
    bars = select_bar_source(settings, client)
    news = select_news_source(settings, client)

    assert isinstance(bars, MassivePriceBarSource)
    assert isinstance(news, FinnhubNewsSource)
    assert isinstance(bars, PriceBarSource)  # and still satisfies the seam
    assert isinstance(news, NewsSource)


async def test_demo_mode_selects_offline_stand_ins(client: httpx.AsyncClient) -> None:
    settings = _settings(demo_mode=True)
    bars = select_bar_source(settings, client)
    news = select_news_source(settings, client)

    assert isinstance(bars, SeedBarSource)  # committed CSV carries the demo anomaly
    assert isinstance(news, NullNewsSource)  # no live Finnhub call, no key needed
    assert isinstance(bars, PriceBarSource)
    assert isinstance(news, NewsSource)
