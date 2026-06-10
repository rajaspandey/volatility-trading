"""One-time script to download and cache all raw market data."""

import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import MarketDataPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(force: bool = False) -> None:
    root = Path(__file__).parent.parent
    with open(root / "config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open(root / "config/strategy_params.yaml") as f:
        params = yaml.safe_load(f)

    loader = DataLoader(settings)
    loader.fetch_all(force=force)

    daily_raw = loader.load_daily_raw()
    hourly_raw = loader.load_hourly_raw()
    fred_rate = loader.load_fred_rate()

    prep = MarketDataPreprocessor(settings, params)
    daily, hourly = prep.build(daily_raw, hourly_raw, fred_rate)

    print(f"\nDaily data: {len(daily)} rows  ({daily.index[0].date()} → {daily.index[-1].date()})")
    print(f"Hourly data: {len(hourly)} rows")
    print(f"\nDaily columns: {list(daily.columns)}")
    print("\nSample daily row (last):")
    print(daily.iloc[-1].to_string())


if __name__ == "__main__":
    force = "--force" in sys.argv
    main(force=force)
