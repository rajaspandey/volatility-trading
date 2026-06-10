"""RiskManager: stop-loss and delta-exit checks."""

from __future__ import annotations

import logging

import pandas as pd

from src.portfolio.portfolio import Portfolio
from src.portfolio.hedge import DeltaHedger

logger = logging.getLogger(__name__)


class RiskManager:
    """Enforces two risk rules from the paper.

    1. Stop-loss: Close all positions if portfolio equity drops more than
       `stop_loss_pct` (default 2%) below the peak equity since the last
       monthly reset.

    2. Delta-exit: Close all positions if |Δ(t) − Δ(t−1)| > threshold
       after a hedge event. Applied at portfolio level.

    Both checks run after every hedge event.
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.02,
        delta_exit_threshold: float = 0.75,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.delta_exit_threshold = delta_exit_threshold
        self._peak_equity: float | None = None

    def reset_peak(self, equity: float) -> None:
        """Reset peak equity (call at start of each monthly cycle)."""
        self._peak_equity = equity
        logger.debug("RiskManager: peak equity reset to %.2f", equity)

    def update_peak(self, equity: float) -> None:
        """Ratchet peak up if equity has risen."""
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

    def check(
        self,
        ts: pd.Timestamp,
        portfolio: Portfolio,
        universe,
        surface,
        pricer,
        hedger: DeltaHedger,
    ) -> str | None:
        """Run both risk checks. Returns 'stop_loss', 'delta_exit', or None.

        If a check fires, closes all positions and resets peak equity.
        """
        equity = portfolio.mark_to_market(ts, universe, surface, pricer)
        self.update_peak(equity)

        # 1. Stop-loss
        if self._peak_equity is not None:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown > self.stop_loss_pct:
                logger.warning(
                    "%s STOP-LOSS fired: equity=%.2f, peak=%.2f, dd=%.2f%%",
                    ts, equity, self._peak_equity, drawdown * 100,
                )
                portfolio.close_all(ts, universe, surface, pricer, reason="stop_loss")
                self.reset_peak(portfolio.mark_to_market(ts, universe, surface, pricer))
                return "stop_loss"

        # 2. Delta-exit
        delta_shift = hedger.delta_shift(ts, portfolio, universe, surface, pricer)
        if delta_shift > self.delta_exit_threshold:
            logger.warning(
                "%s DELTA-EXIT fired: |Δ shift|=%.4f > %.4f",
                ts, delta_shift, self.delta_exit_threshold,
            )
            portfolio.close_all(ts, universe, surface, pricer, reason="delta_exit")
            self.reset_peak(portfolio.mark_to_market(ts, universe, surface, pricer))
            return "delta_exit"

        return None
