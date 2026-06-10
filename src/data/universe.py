"""MarketUniverse: central read-only market data query object."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MarketUniverse:
    """Provides point-in-time market data to strategies and the engine.

    All data is loaded into memory once. Lookups use .loc / .asof for
    speed. All timestamps must be timezone-aware (America/New_York for
    hourly; tz-naive for daily date lookups).
    """

    def __init__(self, daily: pd.DataFrame, hourly: pd.DataFrame):
        self._daily = daily.copy()
        self._daily.index = pd.to_datetime(self._daily.index)

        self._hourly = hourly.copy()
        self._hourly.index = pd.to_datetime(self._hourly.index)

        # Pre-build date index for fast daily lookups
        self._dates = self._daily.index.normalize()

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    def spot(self, ts: pd.Timestamp) -> float:
        """SPY closing price at or before ts."""
        return float(self._get_hourly_or_daily(ts, "close" if "close" in self._hourly.columns else "spy"))

    def rate(self, ts: pd.Timestamp) -> float:
        """Annualized risk-free rate."""
        return float(self._get_daily(ts, "rf_rate"))

    def iv_atm_30d(self, ts: pd.Timestamp) -> float:
        """30-day ATM IV (VIX / 100)."""
        return float(self._get_daily(ts, "iv_30"))

    def iv_atm_5d(self, ts: pd.Timestamp) -> float:
        """5-day ATM IV (VIX9D / 100)."""
        return float(self._get_daily(ts, "iv_5"))

    def skew_index(self, ts: pd.Timestamp) -> float:
        """Raw CBOE SKEW index value."""
        return float(self._get_daily(ts, "skew"))

    def rv(self, ts: pd.Timestamp, window: int) -> float:
        """Rolling realized volatility for the given window (5, 30, or 60)."""
        col = f"rv_{window}"
        if col not in self._daily.columns:
            raise ValueError(f"rv_{window} not in daily data; available: {[c for c in self._daily.columns if c.startswith('rv_')]}")
        return float(self._get_daily(ts, col))

    def iv_rank(self, ts: pd.Timestamp, tenor: str = "30") -> float:
        """IV rank for the given tenor ('5' or '30')."""
        return float(self._get_daily(ts, f"iv_rank_{tenor}"))

    def signal(self, ts: pd.Timestamp, col: str) -> float:
        """Generic accessor for any daily column."""
        return float(self._get_daily(ts, col))

    def daily_row(self, dt: pd.Timestamp) -> pd.Series:
        """Full daily row for a given date."""
        d = pd.Timestamp(dt)
        d = d.tz_localize(None) if d.tzinfo is not None else d
        d = d.normalize()
        idx = self._daily.index.searchsorted(d, side="right") - 1
        if idx < 0:
            raise KeyError(f"No daily data on or before {d}")
        return self._daily.iloc[idx]

    def hourly_row(self, ts: pd.Timestamp) -> pd.Series:
        """Full hourly row at or before ts."""
        if self._hourly.empty:
            return self.daily_row(ts)
        hourly_tz = self._hourly.index.tz
        ts_tz = getattr(ts, "tzinfo", None)
        if (hourly_tz is None) != (ts_tz is None):
            return self.daily_row(ts)
        idx = self._hourly.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            raise KeyError(f"No hourly data on or before {ts}")
        return self._hourly.iloc[idx]

    def trading_days(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """Return trading days (daily index entries) in [start, end]."""
        mask = (self._daily.index >= pd.Timestamp(start)) & (self._daily.index <= pd.Timestamp(end))
        return self._daily.index[mask]

    def hourly_index(self) -> pd.DatetimeIndex:
        return self._hourly.index

    def daily_index(self) -> pd.DatetimeIndex:
        return self._daily.index

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_daily(self, ts: pd.Timestamp, col: str) -> float:
        d = pd.Timestamp(ts).tz_localize(None) if pd.Timestamp(ts).tzinfo is not None else pd.Timestamp(ts)
        d = d.normalize()
        idx = self._daily.index.searchsorted(d, side="right") - 1
        if idx < 0:
            raise KeyError(f"No daily data on or before {d}")
        val = self._daily.iloc[idx][col]
        if pd.isna(val):
            # Search backward for last non-NaN
            for i in range(idx - 1, max(idx - 30, -1), -1):
                v = self._daily.iloc[i][col]
                if not pd.isna(v):
                    return float(v)
        return float(val) if not pd.isna(val) else np.nan

    def _get_hourly_or_daily(self, ts: pd.Timestamp, col: str) -> float:
        """Try hourly first; fall back to daily.

        Handles tz-aware/tz-naive mismatch: if ts is tz-naive and the hourly
        index is tz-aware, skip hourly lookup and use daily directly.
        """
        if not self._hourly.empty and col in self._hourly.columns:
            hourly_tz = self._hourly.index.tz
            ts_tz = getattr(ts, "tzinfo", None)
            compatible = (hourly_tz is None) == (ts_tz is None)
            if compatible:
                idx = self._hourly.index.searchsorted(pd.Timestamp(ts), side="right") - 1
                if idx >= 0:
                    return float(self._hourly.iloc[idx][col])
        daily_col = "spy" if col == "close" else col
        return self._get_daily(ts, daily_col)
