"""WeeklyStrangleStrategy: sell short-dated strangle every trading day."""

from __future__ import annotations

import math
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.options.position import OptionPosition, CONTRACT_UNIT

STRANGLE_T = 5 / 252.0   # 5 trading days
ROUND_TICK = 0.50         # round strikes to nearest $0.50


def _round_tick(x: float, tick: float = 0.50) -> float:
    return round(x / tick) * tick


class WeeklyStrangleStrategy(BaseStrategy):
    """Sell a new 5-day strangle every morning.

    Notional per trade = portfolio_equity / 5.
    K_put  = S − ss × S × σ₆₀
    K_call = S + ss × S × σ₆₀
    where ss = strangle_size ∈ {0, 0.05, 0.1}.

    When ss = 0 both legs are ATM (same K as spot).
    """

    def __init__(self, strangle_size: float = 0.1):
        self.strangle_size = strangle_size

    def on_open(self, ts, portfolio, universe, surface, pricer) -> List[OptionPosition]:
        S = universe.spot(ts)
        r = universe.rate(ts)
        rv60 = universe.rv(ts, window=60)

        open_dt   = pd.Timestamp(ts).date()
        expiry_dt = open_dt + timedelta(days=7)  # ~5 trading days

        # Use actual calendar days to expiry so open_price matches MTM pricing
        T = max((pd.Timestamp(expiry_dt) - pd.Timestamp(ts)).days / 252.0, 1/252.0)

        width = self.strangle_size * S * rv60
        K_put  = _round_tick(S - width)
        K_call = _round_tick(S + width)

        # Clamp strikes to positive values
        K_put  = max(K_put,  1.0)
        K_call = max(K_call, K_put + ROUND_TICK)

        put_sigma  = surface.iv_for_strike(ts, K_put,  T, "p")
        call_sigma = surface.iv_for_strike(ts, K_call, T, "c")
        put_price  = pricer.price("p", S, K_put,  T, r, put_sigma)
        call_price = pricer.price("c", S, K_call, T, r, call_sigma)

        equity = portfolio.mark_to_market(ts, universe, surface, pricer)
        notional_per_trade = equity / 5.0
        n_contracts = max(1, math.floor(notional_per_trade / (S * CONTRACT_UNIT)))

        put_leg = OptionPosition(
            flag="p", K=float(K_put),
            open_date=open_dt, expiry_date=expiry_dt,
            quantity=-n_contracts,
            open_price=float(put_price),
            open_sigma=float(put_sigma),
            notional=float(notional_per_trade),
        )
        call_leg = OptionPosition(
            flag="c", K=float(K_call),
            open_date=open_dt, expiry_date=expiry_dt,
            quantity=-n_contracts,
            open_price=float(call_price),
            open_sigma=float(call_sigma),
            notional=float(notional_per_trade),
        )
        return [put_leg, call_leg]
