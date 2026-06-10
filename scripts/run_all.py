#!/usr/bin/env python
"""Run all four strategies for the paper's Jan-Nov 2022 benchmark and save results."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open("config/strategy_params.yaml") as f:
        params = yaml.safe_load(f)

    cap = settings["portfolio"]["initial_capital"]
    backtest_start = settings["data"]["backtest_start"]
    backtest_end   = "2022-11-30"   # paper benchmark period

    # Load data
    from src.data.universe import MarketUniverse
    daily  = pd.read_parquet("data/processed/market_daily.parquet")
    hourly = pd.read_parquet("data/processed/market_hourly.parquet")
    universe = MarketUniverse(daily, hourly)

    from src.engine.backtester import Backtester
    from src.reporting.metrics import compute_metrics

    bt = Backtester(universe, settings)
    all_results = {}

    # --- Strategy 1: Short Straddle ---
    print("Running Short Straddle …")
    from src.strategies.straddle import ShortStraddleStrategy
    r1 = bt.run(ShortStraddleStrategy(), start=backtest_start, end=backtest_end)
    all_results["straddle"] = r1
    m = compute_metrics(r1["equity_curve"], cap)
    print(f"  Straddle: ret={m['ann_return']:+.1%}, Sharpe={m['sharpe']:.2f}, MaxDD={m['max_drawdown']:.1%}")
    print(f"  Paper OS-A: -3.3% / -0.58")

    # --- Strategy 2: Weekly Strangle (3 sizes) ---
    from src.strategies.strangle import WeeklyStrangleStrategy
    for ss in params["strangle"]["sizes"]:
        print(f"Running Strangle ss={ss} …")
        key = f"strangle_ss{int(ss*100)}"
        r = bt.run(WeeklyStrangleStrategy(ss), start=backtest_start, end=backtest_end)
        all_results[key] = r
        m = compute_metrics(r["equity_curve"], cap)
        print(f"  Strangle(ss={ss}): ret={m['ann_return']:+.1%}, Sharpe={m['sharpe']:.2f}")
    print(f"  Paper OS-A ss=0.1: +2.94% / 0.79")

    # --- Strategy 3: Weighted Skew Arb (warm-up from data start) ---
    print("Running Skew Arb (with 90-day warm-up) …")
    data_start = settings["data"]["start_date"]
    from src.strategies.skew_arb import WeightedSkewArbStrategy
    skew_strategy = WeightedSkewArbStrategy(
        lambda_lookback=params["skew_arb"]["lambda_lookback_days"],
    )
    r3 = bt.run(skew_strategy, start=data_start, end=backtest_end)
    all_results["skew_arb"] = r3
    eq3_22 = r3["equity_curve"][r3["equity_curve"].index >= backtest_start]
    m3 = compute_metrics(eq3_22, cap)
    print(f"  Skew Arb (2022): ret={m3['ann_return']:+.1%}, Sharpe={m3['sharpe']:.2f}, MaxDD={m3['max_drawdown']:.1%}")
    print(f"  Paper OS-A: +11.5% / 0.93")

    # Save equity curves
    out_dir = Path("data/processed")
    for name, res in all_results.items():
        eq = res["equity_curve"]
        if not eq.empty:
            eq.to_parquet(out_dir / f"equity_{name}.parquet")

    print("\nAll equity curves saved to data/processed/equity_*.parquet")

    # Print comparison table
    print("\n=== Jan-Nov 2022 Performance Summary ===")
    print(f"{'Strategy':<25} {'Ann Ret':>9} {'Sharpe':>8} {'Max DD':>8}")
    print("-" * 53)
    for name, res in all_results.items():
        eq = res["equity_curve"]
        if "2022" in name.lower() or name == "skew_arb":
            eq_slice = eq[eq.index >= backtest_start] if name == "skew_arb" else eq
        else:
            eq_slice = eq
        if eq_slice.empty:
            continue
        m = compute_metrics(eq_slice, cap)
        print(f"{name:<25} {m['ann_return']:>+9.2%} {m['sharpe']:>8.2f} {m['max_drawdown']:>8.2%}")


if __name__ == "__main__":
    main()
