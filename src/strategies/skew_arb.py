"""WeightedSkewArbStrategy: buy ATM put, sell λ OTM puts (monthly).

λ = rolling 90-day mean of Γ_ATM(K1) / Γ_OTM(K2).
Positions sized at 2× portfolio notional (2× leverage per paper).
"""

from __future__ import annotations

import math
from collections import deque
from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.options.position import OptionPosition, CONTRACT_UNIT

OTM_RATIO   = 0.97        # K2 = K1 × 0.97
LEVERAGE    = 2.0
LAMBDA_LOOKBACK = 90      # trading days


def _third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    days_to_friday = (4 - d.weekday()) % 7
    first_friday = d + timedelta(days=days_to_friday)
    return first_friday + timedelta(weeks=2)


def _nearest_monthly_expiry(open_date: date) -> date:
    year, month = open_date.year, open_date.month
    candidate = _third_friday(year, month)
    if candidate <= open_date:
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
        candidate = _third_friday(year, month)
    return candidate


class WeightedSkewArbStrategy(BaseStrategy):
    """Monthly skew arbitrage: long ATM put, short λ OTM puts.

    Cold start: first trade requires 90 trading days of gamma history
    (approximately April 2022 if backtest starts 2022-01-01).
    """

    def __init__(self, lambda_lookback: int = LAMBDA_LOOKBACK):
        self._lookback = lambda_lookback
        self._gamma_ratios: deque = deque(maxlen=lambda_lookback)
        self._last_open_month: int | None = None
        self._lambda: float | None = None

    def _update_gamma_history(self, ts, S, r, T, surface, pricer) -> None:
        """Record today's Γ_ATM / Γ_OTM ratio for λ computation."""
        K1 = round(S)
        K2 = round(K1 * OTM_RATIO)

        sigma_atm = surface.iv_for_strike(ts, K1, T, "p")
        sigma_otm = surface.iv_for_strike(ts, K2, T, "p")

        gamma_atm = pricer.gamma(S, K1, T, r, sigma_atm)
        gamma_otm = pricer.gamma(S, K2, T, r, sigma_otm)

        if gamma_otm > 1e-10:
            self._gamma_ratios.append(gamma_atm / gamma_otm)

    def on_open(self, ts, portfolio, universe, surface, pricer) -> List[OptionPosition]:
        S = universe.spot(ts)
        r = universe.rate(ts)

        open_dt_for_gamma = pd.Timestamp(ts).date()
        expiry_for_gamma  = _nearest_monthly_expiry(open_dt_for_gamma)
        T = max((pd.Timestamp(expiry_for_gamma) - pd.Timestamp(ts)).days / 252.0, 1/252.0)

        # Always accumulate gamma history
        self._update_gamma_history(ts, S, r, T, surface, pricer)

        month = pd.Timestamp(ts).month
        if month == self._last_open_month:
            return []

        # Cold start: need full lookback window before first trade
        if len(self._gamma_ratios) < self._lookback:
            self._last_open_month = month
            return []

        self._lambda = float(np.mean(self._gamma_ratios))
        lam = self._lambda

        K1 = float(round(S))
        K2 = float(round(K1 * OTM_RATIO))

        open_dt_actual = pd.Timestamp(ts).date()
        expiry_dt_temp = _nearest_monthly_expiry(open_dt_actual)
        # Use actual calendar T so open_price matches MTM pricing
        T_actual = max((pd.Timestamp(expiry_dt_temp) - pd.Timestamp(ts)).days / 252.0, 1/252.0)

        sigma_atm = surface.iv_for_strike(ts, K1, T_actual, "p")
        sigma_otm = surface.iv_for_strike(ts, K2, T_actual, "p")
        price_atm = pricer.price("p", S, K1, T_actual, r, sigma_atm)
        price_otm = pricer.price("p", S, K2, T_actual, r, sigma_otm)

        equity = portfolio.mark_to_market(ts, universe, surface, pricer)
        notional = equity * LEVERAGE
        n_atm = max(1, math.floor(notional / (S * CONTRACT_UNIT)))
        n_otm = max(1, round(n_atm * lam))

        open_dt   = pd.Timestamp(ts).date()
        expiry_dt = _nearest_monthly_expiry(open_dt)

        atm_leg = OptionPosition(
            flag="p", K=K1,
            open_date=open_dt, expiry_date=expiry_dt,
            quantity=n_atm,       # LONG
            open_price=float(price_atm),
            open_sigma=float(sigma_atm),
            notional=float(notional),
        )
        otm_leg = OptionPosition(
            flag="p", K=K2,
            open_date=open_dt, expiry_date=expiry_dt,
            quantity=-n_otm,      # SHORT
            open_price=float(price_otm),
            open_sigma=float(sigma_otm),
            notional=float(notional),
        )

        self._last_open_month = month
        return [atm_leg, otm_leg]

    def on_risk_exit(self, ts) -> None:
        self._last_open_month = None

    @property
    def current_lambda(self) -> float | None:
        return self._lambda
