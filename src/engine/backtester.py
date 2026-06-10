"""Backtester: main event loop."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from src.engine.clock import TradingClock
from src.engine.events import (
    OpenPositionEvent,
    HedgeEvent,
    RiskCheckEvent,
    ExpiryCheckEvent,
    EODEvent,
)
from src.portfolio.portfolio import Portfolio
from src.portfolio.hedge import DeltaHedger
from src.portfolio.risk import RiskManager
from src.options.pricing import OptionPricer
from src.options.iv_surface import IVSurface

logger = logging.getLogger(__name__)


class Backtester:
    """Drives the event loop for one or more strategies.

    Usage:
        bt = Backtester(universe, settings)
        result = bt.run(strategy, start="2022-01-01", end="2024-12-31")
    """

    def __init__(self, universe, settings: dict):
        self._universe = universe
        self._settings = settings
        self._pricer = OptionPricer()
        self._surface = IVSurface(
            universe,
            skew_scale=settings.get("strategy", {}).get("skew_scale", 0.3),
        )

    def run(self, strategy, start: str, end: str) -> dict:
        """Run a single strategy backtest. Returns results dict."""
        capital = self._settings["portfolio"]["initial_capital"]
        stop_loss_pct = self._settings["risk"]["stop_loss_pct"]
        delta_exit_threshold = self._settings["risk"]["delta_exit_threshold"]

        portfolio = Portfolio(initial_capital=capital)
        hedger = DeltaHedger()
        risk_mgr = RiskManager(
            stop_loss_pct=stop_loss_pct,
            delta_exit_threshold=delta_exit_threshold,
        )
        risk_mgr.reset_peak(capital)

        clock = TradingClock(
            hourly=self._universe._hourly,
            daily=self._universe._daily,
        )

        equity_curve: List[dict] = []
        risk_events: List[dict] = []
        _first_monthly_reset = True

        for event in clock.events(start, end):
            ts = event.ts

            if isinstance(event, OpenPositionEvent):
                # Strategies open new positions at 10AM
                new_positions = strategy.on_open(
                    ts, portfolio, self._universe, self._surface, self._pricer
                )
                for pos in new_positions:
                    portfolio.open_position(pos)
                    logger.debug("%s opened %s K=%.0f q=%d", ts, pos.flag, pos.K, pos.quantity)

                # Monthly reset of risk peak on first trade day
                if new_positions and _first_monthly_reset:
                    risk_mgr.reset_peak(portfolio.mark_to_market(
                        ts, self._universe, self._surface, self._pricer
                    ))
                    _first_monthly_reset = False

            elif isinstance(event, HedgeEvent):
                hedger.hedge(ts, portfolio, self._universe, self._surface, self._pricer)

            elif isinstance(event, RiskCheckEvent):
                fired = risk_mgr.check(
                    ts, portfolio, self._universe, self._surface, self._pricer, hedger
                )
                if fired:
                    risk_events.append({"ts": ts, "type": fired})
                    logger.warning("%s risk event: %s", ts, fired)
                    # Let strategy know (for monthly-cycle strategies that reset lambda etc.)
                    if hasattr(strategy, "on_risk_exit"):
                        strategy.on_risk_exit(ts)

            elif isinstance(event, ExpiryCheckEvent):
                expired_ids = [
                    pid for pid, pos in list(portfolio.positions.items())
                    if pos.is_expired(ts)
                ]
                for pid in expired_ids:
                    pos = portfolio.positions[pid]
                    close_px = pos.current_price(
                        ts, self._universe, self._surface, self._pricer
                    )
                    portfolio.close_position(pid, close_px, ts, reason="expiry")
                    logger.debug("%s expired %s K=%.0f", ts, pos.flag, pos.K)
                # Unwind hedge if no positions left
                if not portfolio.positions and portfolio.hedge_shares != 0.0:
                    spot = self._universe.spot(ts)
                    portfolio.update_hedge(0.0, spot)

            elif isinstance(event, EODEvent):
                equity = portfolio.mark_to_market(
                    ts, self._universe, self._surface, self._pricer
                )
                risk_mgr.update_peak(equity)
                portfolio.log_daily_pnl(
                    ts, self._universe, self._surface, self._pricer
                )
                equity_curve.append({
                    "date": ts,
                    "equity": equity,
                    "cash": portfolio.cash,
                    "n_positions": len(portfolio.positions),
                    "hedge_shares": portfolio.hedge_shares,
                })

        equity_df = pd.DataFrame(equity_curve).set_index("date") if equity_curve else pd.DataFrame()
        return {
            "equity_curve": equity_df,
            "trade_log": portfolio.trade_log,
            "risk_events": pd.DataFrame(risk_events) if risk_events else pd.DataFrame(),
            "final_equity": equity_df["equity"].iloc[-1] if not equity_df.empty else capital,
            "strategy_name": strategy.__class__.__name__,
        }
