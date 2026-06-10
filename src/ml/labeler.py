"""TradeLabeler: assigns y1/y2 labels to per-trade P&L from the trade log."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_trade_sharpe(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Compute per-trade Sharpe from daily PnL in the trade log.

    trade_log columns: ts, position_id, daily_pnl, open_date, expiry_date, ...

    For each position_id, computes:
        mean(daily_pnl) / std(daily_pnl)  over the 5-day life of the strangle.

    Returns DataFrame indexed by position_id with columns:
        open_date, expiry_date, mean_pnl, std_pnl, trade_sharpe
    """
    if trade_log.empty:
        return pd.DataFrame()

    groups = []
    for pid, grp in trade_log.groupby("position_id"):
        pnl = grp["daily_pnl"].values
        mean_p = np.mean(pnl)
        std_p  = np.std(pnl)
        sharpe = mean_p / std_p if std_p > 1e-10 else 0.0
        groups.append({
            "position_id": pid,
            "open_date":   grp["open_date"].iloc[0] if "open_date" in grp.columns else None,
            "expiry_date": grp["expiry_date"].iloc[0] if "expiry_date" in grp.columns else None,
            "ts_open":     grp["ts"].min(),
            "mean_pnl":    mean_p,
            "std_pnl":     std_p,
            "trade_sharpe": sharpe,
        })

    return pd.DataFrame(groups).set_index("position_id")


def label_trades(
    trade_sharpes: pd.DataFrame,
    y1_threshold: float = -0.25,
    y2_threshold: float = 0.30,
) -> pd.DataFrame:
    """Assign binary labels to each trade.

    y1 = 1 if trade_sharpe < y1_threshold (bad trade, avoid)
    y2 = 1 if trade_sharpe > y2_threshold (good trade, size up)
    """
    df = trade_sharpes.copy()
    df["y1"] = (df["trade_sharpe"] < y1_threshold).astype(int)
    df["y2"] = (df["trade_sharpe"] > y2_threshold).astype(int)
    return df


def build_labeled_dataset(
    trade_log: pd.DataFrame,
    features: pd.DataFrame,
    y1_threshold: float = -0.25,
    y2_threshold: float = 0.30,
) -> pd.DataFrame:
    """Merge per-trade labels with features from the trade's open date.

    Returns a DataFrame with features + y1, y2 columns, indexed by ts_open.
    """
    if trade_log.empty or features.empty:
        return pd.DataFrame()

    trade_sharpes = compute_trade_sharpe(trade_log)
    labeled = label_trades(trade_sharpes, y1_threshold, y2_threshold)

    # Join features on the date the trade opened
    labeled["ts_open"] = pd.to_datetime(labeled["ts_open"])
    features.index = pd.to_datetime(features.index)

    result = labeled.join(
        features, on="ts_open", how="inner", rsuffix="_feat"
    )
    return result.dropna(subset=list(features.columns[:10]))   # drop rows with no feature data
