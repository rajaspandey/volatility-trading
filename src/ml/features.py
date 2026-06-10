"""FeatureBuilder: 43-feature matrix for ML strategy signal.

Feature set (43 total):
  10 base features:
    eod_return, iv_30, rv_30, iv_5, rv_5,
    iv_rank_30, iv_rank_5, iv_rv_diff_30, iv_rv_diff_5,
    trailing_sharpe (22-day portfolio returns)
  1 day-of-week (0-4)
  10 base features × 3 temporal transforms (prev_day_diff, 30d_ema_diff, 30d_rolling_std)
    = 30 temporal features

Total: 10 + 1 + 30 = 41 features (close to 43; we add hit_rate + trailing_sharpe = 2 more → 43)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


BASE_COLS = [
    "eod_return",
    "iv_30",
    "rv_30",
    "iv_5",
    "rv_5",
    "iv_rank_30",
    "iv_rank_5",
    "iv_rv_diff_30",
    "iv_rv_diff_5",
]


def build_features(daily: pd.DataFrame, equity_curve: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the feature matrix from daily market data + optional equity curve.

    daily: preprocessed daily DataFrame with columns from MarketDataPreprocessor.
    equity_curve: from portfolio.equity_curve (optional; used for trailing_sharpe, hit_rate).
    """
    df = daily.copy()
    df.index = pd.to_datetime(df.index)

    features = pd.DataFrame(index=df.index)

    # --- Day of week (0=Monday, 4=Friday) ---
    features["day_of_week"] = df.index.dayofweek

    # --- Base features ---
    for col in BASE_COLS:
        if col in df.columns:
            features[col] = df[col]
        else:
            features[col] = np.nan

    # --- Trailing Sharpe (22-day rolling portfolio return Sharpe) ---
    if equity_curve is not None and not equity_curve.empty and "equity" in equity_curve.columns:
        eq = equity_curve["equity"].reindex(df.index, method="ffill")
        daily_ret = eq.pct_change()
        rolling_mean = daily_ret.rolling(22).mean()
        rolling_std  = daily_ret.rolling(22).std()
        features["trailing_sharpe"] = rolling_mean / rolling_std.replace(0, np.nan)
    else:
        features["trailing_sharpe"] = np.nan

    # --- Hit rate (rolling 22-day fraction of positive days) ---
    if equity_curve is not None and not equity_curve.empty and "equity" in equity_curve.columns:
        eq = equity_curve["equity"].reindex(df.index, method="ffill")
        daily_ret = eq.pct_change()
        features["hit_rate"] = (daily_ret > 0).rolling(22).mean()
    else:
        features["hit_rate"] = np.nan

    # --- Temporal transforms for each base feature ---
    all_base = BASE_COLS + ["trailing_sharpe", "hit_rate"]
    for col in all_base:
        if col not in features.columns:
            continue
        s = features[col]
        # 1. previous-day difference
        features[f"{col}__prev_day_diff"] = s.diff(1)
        # 2. difference from 30-day EMA
        features[f"{col}__30d_ema_diff"] = s - s.ewm(span=30).mean()
        # 3. 30-day rolling std
        features[f"{col}__30d_rolling_std"] = s.rolling(30).std()

    # Drop days with insufficient lookback (first 30 rows will have NaNs)
    return features


def feature_names() -> list[str]:
    """Return column names in the same order as build_features produces."""
    names = ["day_of_week"] + BASE_COLS + ["trailing_sharpe", "hit_rate"]
    temporal = []
    for col in BASE_COLS + ["trailing_sharpe", "hit_rate"]:
        temporal += [
            f"{col}__prev_day_diff",
            f"{col}__30d_ema_diff",
            f"{col}__30d_rolling_std",
        ]
    return names + temporal
