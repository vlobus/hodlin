"""Seed price-bar connector (``PriceBarSource``) — a committed CSV (D13).

A deterministic, offline fallback and cold-start backfill source: it satisfies
the same Protocol as the live providers but reads bars from a bundled CSV, so
the demo runs on a clean machine with no network and no keys. Later DVC-tracked.

The requested ``[start, end]`` window is ignored: this is a fixed historical
fixture (real June-2024 bars carrying the demo anomaly) standing in for
"whatever recent history you asked for". Honouring the window would make the
demo depend on the wall clock — the scheduled backfill asks for the last N
days, which the fixed dates would never fall inside. The live providers
(Massive) do honour the window; only this stand-in returns its whole series.
"""

import csv
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from hodlin_recommend.domain.models import PriceBar

_CSV_PATH = Path(__file__).with_name("seed_bars.csv")


class SeedBarSource:
    source = "seed"

    def __init__(self, csv_path: Path | None = None) -> None:
        self._csv_path = csv_path or _CSV_PATH

    async def get_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[PriceBar]:
        # start/end are accepted to satisfy the Protocol but deliberately
        # ignored — see the module docstring.
        bars: list[PriceBar] = []
        with self._csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["symbol"] != symbol or row["interval"] != interval:
                    continue
                ts = datetime.fromisoformat(row["ts"]).astimezone(UTC)
                volume = row.get("volume") or None
                bars.append(
                    PriceBar(
                        symbol=symbol,
                        interval=interval,
                        ts=ts,
                        open=Decimal(row["open"]),
                        high=Decimal(row["high"]),
                        low=Decimal(row["low"]),
                        close=Decimal(row["close"]),
                        volume=Decimal(volume) if volume is not None else None,
                        source="seed",
                    )
                )
        bars.sort(key=lambda bar: bar.ts)
        return bars

    async def health(self) -> bool:
        return self._csv_path.exists()
