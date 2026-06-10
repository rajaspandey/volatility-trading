"""ShortStraddleStrategy: sell ATM straddle at start of each month."""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import List

import pandas as pd

from src.strategies.base import BaseStrategy
from src.options.position import OptionPosition, CONTRACT_UNIT


def _third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of the given year/month."""
    d = date(year, month, 1)
    # Find first Friday
    days_to_friday = (4 - d.weekday()) % 7
    first_friday = d + timedelta(days=days_to_friday)
    return first_friday + timedelta(weeks=2)


def _nearest_monthly_expiry(open_date: date) -> date:
    """Return the 3rd Friday of the following month (≈21 trading days)."""
    year, month = open_date.year, open_date.month
    candidate = _third_friday(year, month)
    if candidate <= open_date:
        # Already past this month's expiry; use next month
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
        candidate = _third_friday(year, month)
    return candidate


class ShortStraddleStrategy(BaseStrategy):
    """Sell 1 ATM call + 1 ATM put on the first trading day of each month.

    Position sizes: floor(equity / (S × 100)) contracts each leg.
    Expiry: 3rd Friday of the following month.
    The backtester closes positions at expiry automatically.
    """

    def __init__(self):
        self._last_open_month: int | None = None

    def on_open(self, ts, portfolio, universe, surface, pricer) -> List[OptionPosition]:
        month = ts.month if hasattr(ts, "month") else pd.Timestamp(ts).month
        if month == self._last_open_month:
            return []

        S = universe.spot(ts)
        r = universe.rate(ts)

        open_dt   = pd.Timestamp(ts).date()
        expiry_dt = _nearest_monthly_expiry(open_dt)

        # Use actual calendar days to expiry so open_price matches MTM pricing
        T = max((pd.Timestamp(expiry_dt) - pd.Timestamp(ts)).days / 252.0, 1/252.0)
        K = round(S)   # ATM strike

        call_sigma = surface.iv_for_strike(ts, K, T, "c")
        put_sigma  = surface.iv_for_strike(ts, K, T, "p")
        call_price = pricer.price("c", S, K, T, r, call_sigma)
        put_price  = pricer.price("p", S, K, T, r, put_sigma)

        equity = portfolio.mark_to_market(ts, universe, surface, pricer)
        n_contracts = max(1, math.floor(equity / (S * CONTRACT_UNIT)))

        call_leg = OptionPosition(
            flag="c", K=float(K),
            open_date=open_dt, expiry_date=expiry_dt,
            quantity=-n_contracts,
            open_price=float(call_price),
            open_sigma=float(call_sigma),
            notional=float(S * n_contracts * CONTRACT_UNIT),
        )
        put_leg = OptionPosition(
            flag="p", K=float(K),
            open_date=open_dt, expiry_date=expiry_dt,
            quantity=-n_contracts,
            open_price=float(put_price),
            open_sigma=float(put_sigma),
            notional=float(S * n_contracts * CONTRACT_UNIT),
        )

        self._last_open_month = month
        return [call_leg, put_leg]

    def on_risk_exit(self, ts) -> None:
        # Reset so next trading day we re-enter
        self._last_open_month = None
