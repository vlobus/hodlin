"""Anomaly detector tests — pure math, no I/O.

Covers the T5 acceptance: z-scores match hand-computed values, including the
edge cases (sigma~0, too few bars, up and down directions), and the committed
seed data yields the demo anomaly without a database or network.

The hand-built series alternates closes 100/101 so its four baseline log
returns are exactly [+ln 1.01, -ln 1.01, +ln 1.01, -ln 1.01]: mean 0, sample
stddev ln(1.01) * sqrt(4/3). The expected z is then the closed form
final_return / (ln(1.01) * sqrt(4/3)), computed independently of the code.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from hodlin_recommend.connectors.seed_bars import SeedBarSource
from hodlin_recommend.domain.anomaly import detect_latest, detect_series
from hodlin_recommend.domain.models import PriceBar

_WINDOW = 4
_UP_Z = 4.246444  # ln(1.05) / (ln(1.01) * sqrt(4/3))
_DOWN_Z = -4.464303  # ln(0.95) / (ln(1.01) * sqrt(4/3))


def _bars(closes: list[str], symbol: str = "TEST") -> list[PriceBar]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        PriceBar(
            symbol=symbol,
            interval="1d",
            ts=start + timedelta(days=i),
            open=Decimal(close),
            high=Decimal(close),
            low=Decimal(close),
            close=Decimal(close),
            source="test",
        )
        for i, close in enumerate(closes)
    ]


_CALM = ["100", "101", "100", "101", "100"]  # zero-mean baseline, sd = ln(1.01)*sqrt(4/3)


def test_up_spike_matches_hand_computed_z() -> None:
    anomaly = detect_latest(_bars([*_CALM, "105"]), window=_WINDOW, threshold=2.5)

    assert anomaly is not None
    assert float(anomaly.z_score) == pytest.approx(_UP_Z, abs=1e-6)
    assert anomaly.direction == "up"
    assert anomaly.return_pct == Decimal("5")  # 100 -> 105 exactly, no float drift
    assert anomaly.window == _WINDOW
    assert anomaly.bar_ts == datetime(2026, 1, 6, tzinfo=UTC)


def test_down_spike_matches_hand_computed_z() -> None:
    anomaly = detect_latest(_bars([*_CALM, "95"]), window=_WINDOW, threshold=2.5)

    assert anomaly is not None
    assert float(anomaly.z_score) == pytest.approx(_DOWN_Z, abs=1e-6)
    assert anomaly.direction == "down"
    assert anomaly.return_pct == Decimal("-5")


def test_below_threshold_is_no_signal() -> None:
    # Same spike, threshold above its |z| -> quiet.
    assert detect_latest(_bars([*_CALM, "105"]), window=_WINDOW, threshold=5.0) is None


def test_flat_baseline_is_no_signal_not_a_crash() -> None:
    # sigma ~ 0 (constant closes) would divide by zero; must be None.
    assert detect_latest(_bars(["100"] * 5 + ["110"]), window=_WINDOW, threshold=2.5) is None


def test_too_few_bars_is_no_signal() -> None:
    # window=4 needs 4 baseline returns + the scored one = 6 bars; 5 isn't enough.
    five_bars = _bars(["100", "101", "100", "101", "110"])
    assert detect_latest(five_bars, window=_WINDOW, threshold=2.5) is None
    assert detect_series(five_bars, window=_WINDOW, threshold=2.5) == []
    assert detect_latest([], window=_WINDOW, threshold=2.5) is None


def test_series_finds_mid_history_spike_and_input_order_is_irrelevant() -> None:
    closes = [*_CALM, "105", "105", "106", "105", "106"]
    bars = _bars(closes)
    expected = detect_series(bars, window=_WINDOW, threshold=2.5)

    assert [a.bar_ts for a in expected] == [datetime(2026, 1, 6, tzinfo=UTC)]
    assert float(expected[0].z_score) == pytest.approx(_UP_Z, abs=1e-6)
    assert detect_series(list(reversed(bars)), window=_WINDOW, threshold=2.5) == expected


def test_invalid_inputs_raise() -> None:
    bars = _bars([*_CALM, "105"])
    with pytest.raises(ValueError, match="window"):
        detect_latest(bars, window=1, threshold=2.5)
    mixed = bars + _bars(["100"], symbol="OTHER")
    with pytest.raises(ValueError, match="one symbol"):
        detect_series(mixed, window=_WINDOW, threshold=2.5)


async def test_seed_data_contains_the_demo_anomaly() -> None:
    """The committed CSV must trip the detector at the demo config (window=15,
    threshold=2.5) so the cold-start pass has something to show offline."""
    bars = await SeedBarSource().get_candles(
        "BTC-USD", "1d", datetime(2024, 6, 1, tzinfo=UTC), datetime(2024, 7, 1, tzinfo=UTC)
    )

    anomalies = detect_series(bars, window=15, threshold=2.5)

    assert len(anomalies) >= 1
    assert anomalies[0].direction == "down"
    assert abs(anomalies[0].z_score) >= Decimal("2.5")
