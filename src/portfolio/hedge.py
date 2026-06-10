"""DeltaHedger: compute and execute net-delta hedge trades."""

from __future__ import annotations

import logging

import pandas as pd

from src.portfolio.portfolio import Portfolio

logger = logging.getLogger(__name__)


class DeltaHedger:
    """Computes and executes a delta-neutral hedge for the portfolio.

    Each call to `hedge()` calculates the net portfolio delta (option delta
    plus any existing hedge shares), then trades SPY to bring it to zero.

    Hedge equation (from the paper):
        target_shares = −Σ(Δᵢ × qᵢ × 100)  for all option positions
    which results in net portfolio delta ≈ 0.
    """

    def __init__(self):
        self._last_delta: float = 0.0

    @property
    def last_delta(self) -> float:
        return self._last_delta

    def hedge(
        self,
        ts: pd.Timestamp,
        portfolio: Portfolio,
        universe,
        surface,
        pricer,
    ) -> float:
        """Execute delta hedge; return shares traded (signed).

        If no open positions, unwinds any existing hedge.
        """
        spot = universe.spot(ts)
        if not portfolio.positions:
            if portfolio.hedge_shares != 0.0:
                traded = portfolio.update_hedge(0.0, spot)
                logger.debug("%s hedge unwind: %.1f shares @ %.2f", ts, traded, spot)
            self._last_delta = 0.0
            return 0.0

        # Compute option delta (share-equivalents, excluding current hedge)
        pos_deltas = portfolio.position_deltas(ts, universe, surface, pricer)
        option_delta = sum(pos_deltas.values())

        # Target hedge: offset option delta exactly
        target_hedge = -option_delta
        traded = portfolio.update_hedge(target_hedge, spot)

        # After hedging, net delta ≈ 0; record that as the baseline for drift checks
        self._last_delta = portfolio.net_delta(ts, universe, surface, pricer)

        if abs(traded) > 0.01:
            logger.debug(
                "%s delta hedge: option_Δ=%.3f, traded %.2f shares @ %.2f",
                ts, option_delta, traded, spot,
            )
        return traded

    def delta_shift(
        self,
        ts: pd.Timestamp,
        portfolio: Portfolio,
        universe,
        surface,
        pricer,
    ) -> float:
        """Compute |current_delta - last_delta| without trading (risk check)."""
        current = portfolio.net_delta(ts, universe, surface, pricer)
        shift = abs(current - self._last_delta)
        self._last_delta = current
        return shift
