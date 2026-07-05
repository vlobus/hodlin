"""Rolling z-score anomaly detection on log returns (D6) — pure functions.

The signal is the latest bar's log return measured against the mean/stddev of
the ``window`` returns immediately before it. That baseline self-calibrates per
asset: a 3% day is an anomaly for an index fund and noise for a small-cap
crypto, so a fixed percentage threshold would misfire across volatility
regimes. No I/O here — bars come in, ``Anomaly`` values come out; the caller
(backfill, later the scheduler job) owns fetching and persistence.
"""

import math
from collections.abc import Sequence
from decimal import Decimal
from itertools import pairwise

from hodlin_recommend.domain.models import Anomaly, PriceBar

# Below this the baseline is flat (e.g. a stablecoin or stale feed) and a
# z-score would divide by ~zero; treat as "no signal" rather than +/-inf.
_SIGMA_FLOOR = 1e-12

# Statistics are stored as NUMERIC; six decimal places is far beyond the
# signal's real precision but keeps values readable and comparable.
_Z_PLACES = Decimal("0.000001")
_PCT_PLACES = Decimal("0.0001")


def _validate(bars: Sequence[PriceBar], window: int, threshold: float) -> list[PriceBar]:
    """Reject inputs the math can't honestly answer for; return bars oldest-first."""
    if window < 2:
        raise ValueError(f"window must be >= 2 to compute a sample stddev, got {window}")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    keys = {(bar.symbol, bar.interval) for bar in bars}
    if len(keys) > 1:
        raise ValueError(f"bars must belong to one symbol/interval, got {sorted(keys)}")
    if any(bar.close <= 0 for bar in bars):
        raise ValueError("close prices must be positive to take log returns")
    ordered = sorted(bars, key=lambda bar: bar.ts)
    if any(a.ts == b.ts for a, b in pairwise(ordered)):
        raise ValueError("bars must have distinct timestamps")
    return ordered


def _log_returns(bars: Sequence[PriceBar]) -> list[float]:
    closes = [float(bar.close) for bar in bars]
    return [math.log(after / before) for before, after in pairwise(closes)]


def _zscore(returns: Sequence[float], at: int, window: int) -> float | None:
    """Z-score of ``returns[at]`` against the ``window`` returns before it, or
    ``None`` when there isn't a full baseline or the baseline is flat (sigma~0)."""
    if at < window:
        return None
    baseline = returns[at - window : at]
    mean = sum(baseline) / window
    variance = sum((r - mean) ** 2 for r in baseline) / (window - 1)
    sigma = math.sqrt(variance)
    if sigma < _SIGMA_FLOOR:
        return None
    return (returns[at] - mean) / sigma


def _anomaly(prev: PriceBar, bar: PriceBar, z: float, window: int) -> Anomaly:
    # Percent change is exact Decimal arithmetic on the closes; only the
    # z-score statistic passes through float (it needs log/sqrt anyway).
    return_pct = ((bar.close / prev.close - 1) * 100).quantize(_PCT_PLACES)
    return Anomaly(
        symbol=bar.symbol,
        interval=bar.interval,
        bar_ts=bar.ts,
        z_score=Decimal(f"{z:f}").quantize(_Z_PLACES),
        return_pct=return_pct,
        direction="up" if z > 0 else "down",
        window=window,
    )


def detect_series(bars: Sequence[PriceBar], *, window: int, threshold: float) -> list[Anomaly]:
    """Run the detector over a bar history (any order) and return every bar
    whose |z| >= ``threshold``, oldest first. Used by the cold-start backfill
    so a demo anomaly exists before the first live tick."""
    ordered = _validate(bars, window, threshold)
    returns = _log_returns(ordered)
    anomalies: list[Anomaly] = []
    for at in range(window, len(returns)):
        z = _zscore(returns, at, window)
        if z is not None and abs(z) >= threshold:
            # returns[at] is the move from bar at index ``at`` to ``at + 1``.
            anomalies.append(_anomaly(ordered[at], ordered[at + 1], z, window))
    return anomalies


def detect_latest(bars: Sequence[PriceBar], *, window: int, threshold: float) -> Anomaly | None:
    """Score only the newest bar — the per-tick check the scheduler job runs.
    ``None`` means no signal: too few bars for a full baseline, a flat baseline
    (sigma~0), or |z| below ``threshold``."""
    ordered = _validate(bars, window, threshold)
    returns = _log_returns(ordered)
    if not returns:
        return None
    z = _zscore(returns, len(returns) - 1, window)
    if z is None or abs(z) < threshold:
        return None
    return _anomaly(ordered[-2], ordered[-1], z, window)
