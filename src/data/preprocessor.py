"""MarketDataPreprocessor: builds derived columns from raw data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def _rv(log_returns: pd.Series, window: int) -> pd.Series:
    """Rolling realized volatility (annualized std of log returns)."""
    return log_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def _iv_rank(iv: pd.Series, lookback: int) -> pd.Series:
    """IV rank = (IV - min) / (max - min) over lookback window."""
    mn = iv.rolling(lookback).min()
    mx = iv.rolling(lookback).max()
    denom = mx - mn
    return ((iv - mn) / denom).where(denom > 0, other=np.nan)


class MarketDataPreprocessor:
    """Builds processed parquets with all derived signals."""

    def __init__(self, settings: dict, strategy_params: dict):
        self.settings = settings
        self.params = strategy_params
        self.processed_dir = Path(settings["data"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        daily_raw: pd.DataFrame,
        hourly_raw: pd.DataFrame,
        fred_rate: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build market_daily and market_hourly processed parquets."""
        daily = self._build_daily(daily_raw, fred_rate)
        hourly = self._build_hourly(hourly_raw, daily)
        daily.to_parquet(self.processed_dir / "market_daily.parquet")
        hourly.to_parquet(self.processed_dir / "market_hourly.parquet")
        logger.info("Processed: %d daily rows, %d hourly rows", len(daily), len(hourly))
        return daily, hourly

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        daily = pd.read_parquet(self.processed_dir / "market_daily.parquet")
        hourly = pd.read_parquet(self.processed_dir / "market_hourly.parquet")
        return daily, hourly

    def _build_daily(self, raw: pd.DataFrame, fred_rate: pd.Series) -> pd.DataFrame:
        df = raw.copy()
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = "date"

        spy = df["spy"].ffill()

        # Log returns and realized vols
        log_ret = np.log(spy / spy.shift(1))
        df["log_return"] = log_ret
        df["eod_return"] = spy.pct_change()
        df["rv_5"] = _rv(log_ret, 5)
        df["rv_30"] = _rv(log_ret, 30)
        df["rv_60"] = _rv(log_ret, 60)

        # Implied vols from VIX family (VIX is in percentage points → /100)
        df["iv_30"] = df["vix"].ffill() / 100.0
        df["iv_5"] = df["vix9d"].ffill() / 100.0

        # Fallback: scale 30-day IV to 5-day when VIX9D is missing
        missing_vix9d = df["iv_5"].isna()
        df.loc[missing_vix9d, "iv_5"] = df.loc[missing_vix9d, "iv_30"] * np.sqrt(9 / 30)

        # IV ranks
        df["iv_rank_30"] = _iv_rank(df["iv_30"], lookback=100)
        df["iv_rank_5"] = _iv_rank(df["iv_5"], lookback=22)

        # IV-RV spreads
        df["iv_rv_diff_30"] = df["iv_30"] - df["rv_30"]
        df["iv_rv_diff_5"] = df["iv_5"] - df["rv_5"]

        # SKEW index (forward-fill gaps)
        df["skew"] = df["skew"].ffill()

        # Risk-free rate: prefer FRED, fall back to IRX (/100 since IRX is %)
        if fred_rate is not None and len(fred_rate) > 0:
            rate = fred_rate.reindex(df.index, method="ffill") / 100.0  # already decimal but FRED is %
            # TB3MS comes as percent: 5.25 means 5.25%
            # fred_rate was already divided by 100 in loader → leave as-is
            df["rf_rate"] = fred_rate.reindex(df.index, method="ffill")
        else:
            df["rf_rate"] = df["irx"].ffill() / 100.0

        df["rf_rate"] = df["rf_rate"].ffill().fillna(0.05)

        # SPX close
        if "spx" in df.columns:
            df["spx"] = df["spx"].ffill()

        return df

    def _build_hourly(self, hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
        h = hourly.copy()

        # Align daily signals onto hourly bars (forward-fill)
        daily_cols = [
            "rv_5", "rv_30", "rv_60",
            "iv_30", "iv_5",
            "iv_rank_30", "iv_rank_5",
            "iv_rv_diff_30", "iv_rv_diff_5",
            "skew", "rf_rate", "log_return", "eod_return",
        ]
        daily_indexed = daily[daily_cols].copy()
        daily_indexed.index = pd.to_datetime(daily_indexed.index)

        # Reindex daily onto hourly dates (by date only)
        h["_date"] = h.index.normalize().tz_localize(None)
        daily_indexed.index = daily_indexed.index.tz_localize(None)
        merged = h.join(daily_indexed, on="_date", how="left")
        merged.drop(columns=["_date"], inplace=True)
        merged[daily_cols] = merged[daily_cols].ffill()

        return merged
