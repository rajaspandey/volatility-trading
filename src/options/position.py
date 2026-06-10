"""OptionPosition: immutable record of a single synthetic option contract leg."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

TRADING_DAYS = 252
CONTRACT_UNIT = 100  # shares per standard options contract


@dataclass(frozen=True)
class OptionPosition:
    """Represents a single options leg (call or put), long or short.

    quantity > 0 → long; quantity < 0 → short.
    open_price is the BS price per share at the time of opening.
    """

    flag: str            # 'c' or 'p'
    K: float             # strike price
    open_date: date
    expiry_date: date
    quantity: int        # signed: negative = short
    open_price: float    # BS price per share at open
    open_sigma: float    # IV at open
    notional: float      # dollar notional this leg represents

    position_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # ------------------------------------------------------------------
    # Greeks on demand
    # ------------------------------------------------------------------

    def time_to_expiry(self, ts: pd.Timestamp) -> float:
        """Years remaining to expiry from ts."""
        dt = pd.Timestamp(ts)
        exp = pd.Timestamp(self.expiry_date)
        days = max((exp - dt).days, 0)
        return days / TRADING_DAYS

    def is_expired(self, ts: pd.Timestamp) -> bool:
        return pd.Timestamp(ts).date() >= self.expiry_date

    def current_price(self, ts, universe, surface, pricer) -> float:
        S = universe.spot(ts)
        r = universe.rate(ts)
        T = self.time_to_expiry(ts)
        sigma = surface.iv_for_strike(ts, self.K, T, self.flag)
        return float(pricer.price(self.flag, S, self.K, T, r, sigma))

    def current_delta(self, ts, universe, surface, pricer) -> float:
        S = universe.spot(ts)
        r = universe.rate(ts)
        T = self.time_to_expiry(ts)
        sigma = surface.iv_for_strike(ts, self.K, T, self.flag)
        return float(pricer.delta(self.flag, S, self.K, T, r, sigma))

    def current_gamma(self, ts, universe, surface, pricer) -> float:
        S = universe.spot(ts)
        r = universe.rate(ts)
        T = self.time_to_expiry(ts)
        sigma = surface.iv_for_strike(ts, self.K, T, self.flag)
        return float(pricer.gamma(S, self.K, T, r, sigma))

    def pnl(self, ts, universe, surface, pricer) -> float:
        """Mark-to-market PnL in dollars.

        PnL = (current_price - open_price) × quantity × CONTRACT_UNIT
        For a short position (quantity < 0), price increase → negative PnL.
        """
        curr = self.current_price(ts, universe, surface, pricer)
        return (curr - self.open_price) * self.quantity * CONTRACT_UNIT

    def open_pnl_contribution(self) -> float:
        """Cash received/paid at open (premium × quantity × unit, with sign).

        Selling (quantity < 0): positive cash inflow.
        Buying  (quantity > 0): negative cash outflow.
        """
        return -self.open_price * self.quantity * CONTRACT_UNIT
