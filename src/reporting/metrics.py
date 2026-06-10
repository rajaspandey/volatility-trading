"""PerformanceMetrics: Sharpe ratio, drawdown, annualized return."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(equity_curve: pd.DataFrame, initial_capital: float) -> dict:
    """Compute standard performance metrics from an equity curve DataFrame.

    equity_curve must have a 'equity' column and a DatetimeIndex.
    """
    if equity_curve.empty or "equity" not in equity_curve.columns:
        return _empty_metrics()

    eq = equity_curve["equity"].dropna()
    if len(eq) < 2:
        return _empty_metrics()

    daily_returns = eq.pct_change().dropna()

    total_return = (eq.iloc[-1] / initial_capital) - 1.0
    n_days = len(daily_returns)
    n_years = n_days / 252.0
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 1e-10 else 0.0

    # Max drawdown
    rolling_max = eq.cummax()
    drawdown = (eq - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Win rate (days with positive return)
    win_rate = (daily_returns > 0).mean()

    return {
        "total_return": float(total_return),
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "n_trading_days": int(n_days),
    }


def _empty_metrics() -> dict:
    return {
        "total_return": float("nan"),
        "ann_return": float("nan"),
        "ann_vol": float("nan"),
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "win_rate": float("nan"),
        "n_trading_days": 0,
    }


def summarize_results(results: dict, initial_capital: float) -> pd.Series:
    """Compute metrics from a backtester results dict and return as Series."""
    metrics = compute_metrics(results["equity_curve"], initial_capital)
    metrics["strategy"] = results.get("strategy_name", "Unknown")
    metrics["n_risk_events"] = len(results.get("risk_events", pd.DataFrame()))
    return pd.Series(metrics)


def compare_strategies(results_list: list[dict], initial_capital: float) -> pd.DataFrame:
    """Build a comparison table for multiple strategy results."""
    rows = [summarize_results(r, initial_capital) for r in results_list]
    df = pd.DataFrame(rows).set_index("strategy")
    return df[["ann_return", "sharpe", "max_drawdown", "ann_vol", "win_rate", "n_trading_days", "n_risk_events"]]
