"""Per-asset detection configuration (D6).

The z-score self-calibrates to each asset's volatility, but the window,
threshold, and backfill depth are still judgement calls that differ by asset
class — so they live here per asset, not as one global constant. These are
tuning values, not secrets, so code defaults are appropriate (unlike
``config.py``, where every env value is required).
"""

from pydantic import BaseModel, ConfigDict


class AssetConfig(BaseModel):
    """Everything the ingest/detection pipeline needs to know about one asset."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    kind: str  # "stock" | "crypto"
    name: str | None = None
    interval: str = "1d"
    # Baseline size for the rolling z-score: ~3 trading weeks of daily bars.
    window: int = 15
    # |z| at or above this trips an anomaly. 2.5 sigma keeps daily alerts rare
    # without hiding real moves.
    threshold: float = 2.5
    # Cold-start depth: enough calendar days that the window is full on the
    # first tick even across weekends/holidays.
    backfill_days: int = 30


# The M1 demo universe: one stock, one crypto — enough to show the detector
# calibrating differently per asset. Later this moves to real configuration.
DEFAULT_ASSETS: tuple[AssetConfig, ...] = (
    AssetConfig(symbol="AAPL", kind="stock", name="Apple Inc."),
    AssetConfig(symbol="BTC-USD", kind="crypto", name="Bitcoin / USD"),
)
