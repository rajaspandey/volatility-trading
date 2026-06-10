"""IVSurface: synthetic implied volatility surface from VIX family + SKEW index."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.universe import MarketUniverse


# Tenor anchors for interpolation
VIX9D_DAYS = 9
VIX_DAYS = 30


class IVSurface:
    """Constructs a point-in-time IV surface using freely available data.

    ATM IV:
        30-day → VIX / 100 (definitional: VIX is SPX 30-day ATM IV)
        9-day  → VIX9D / 100
        Other  → linearly interpolated / extrapolated between the two anchors

    OTM skew adjustment:
        Uses the CBOE SKEW index. Higher SKEW (>100) implies fatter left tail
        → OTM puts more expensive than ATM puts.

        Skew model (simplified sticky-strike):
            IV(K) = ATM_IV × (1 + α × log(K/F) / (ATM_IV × sqrt(T)))

        where:
            α = skew_factor derived from SKEW index
            F = S × exp(r × T)  (forward price)

        A SKEW of 130 → α ≈ −0.039 → a put 3% OTM has IV ≈ 2% above ATM.
        skew_scale is a calibration constant (default 0.3) that sets the
        magnitude of this adjustment.
    """

    def __init__(self, universe: MarketUniverse, skew_scale: float = 0.3):
        self._u = universe
        self.skew_scale = skew_scale

    def iv_for_strike(
        self,
        ts: pd.Timestamp,
        K: float,
        T: float,
        flag: str = "p",
    ) -> float:
        """IV for a specific strike K and time-to-expiry T (years).

        For ATM (K ≈ S), returns the interpolated ATM IV.
        For OTM puts (K < S) or OTM calls (K > S), applies a skew adjustment.
        """
        S = self._u.spot(ts)
        r = self._u.rate(ts)
        atm_iv = self.interpolate_atm_iv(ts, T)
        skew_idx = self._u.skew_index(ts)

        if np.isnan(skew_idx) or skew_idx <= 0:
            return atm_iv

        F = S * np.exp(r * T)
        log_moneyness = np.log(K / F)

        # α from SKEW index: SKEW = 100 corresponds to no skew (α = 0)
        # α = -(SKEW - 100) / 1000 × skew_scale
        alpha = -(skew_idx - 100.0) / 1000.0 * self.skew_scale

        denom = atm_iv * np.sqrt(max(T, 1e-6))
        if denom < 1e-10:
            return atm_iv

        skew_adj = alpha * log_moneyness / denom
        iv = atm_iv * (1.0 + skew_adj)

        # Clamp: IV must be positive and sensible
        return float(np.clip(iv, 0.01, 5.0))

    def interpolate_atm_iv(self, ts: pd.Timestamp, T: float) -> float:
        """Linearly interpolate ATM IV between VIX9D (9-day) and VIX (30-day).

        For T outside [9/252, 30/252], extrapolate flat at the nearest anchor.
        """
        iv_9d = self._u.iv_atm_5d(ts)   # VIX9D / 100
        iv_30d = self._u.iv_atm_30d(ts)  # VIX / 100

        T_9 = VIX9D_DAYS / 252.0
        T_30 = VIX_DAYS / 252.0

        if np.isnan(iv_9d):
            iv_9d = iv_30d * np.sqrt(VIX9D_DAYS / VIX_DAYS)

        if T <= T_9:
            return float(iv_9d)
        if T >= T_30:
            return float(iv_30d)

        # Linear interpolation in T-space
        frac = (T - T_9) / (T_30 - T_9)
        return float(iv_9d + frac * (iv_30d - iv_9d))

    def atm_iv(self, ts: pd.Timestamp, T: float) -> float:
        """Convenience alias for interpolate_atm_iv."""
        return self.interpolate_atm_iv(ts, T)
