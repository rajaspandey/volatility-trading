"""Portfolio: tracks positions, cash, hedge shares, and MTM equity."""

from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd

from src.options.position import OptionPosition, CONTRACT_UNIT


class Portfolio:
    """Tracks all open option positions, cash, and delta hedge shares.

    Cash accounting:
        - Opening a position: cash += open_pnl_contribution() (premium received for
          short, premium paid for long)
        - Closing a position: cash += (close_price - open_price) × qty × 100
        - Hedge share trades: cash -= shares_bought × price

    Equity = cash + MTM value of all open positions + hedge_shares × spot
    """

    def __init__(self, initial_capital: float):
        self.cash: float = initial_capital
        self.initial_capital: float = initial_capital
        self.positions: Dict[str, OptionPosition] = {}  # position_id → OptionPosition
        self.hedge_shares: float = 0.0   # net SPY hedge shares (signed)
        self._trade_log: List[dict] = []  # per-position daily PnL rows for ML

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def open_position(self, pos: OptionPosition) -> None:
        """Add a position. Premium is NOT credited to cash at open.
        PnL is realized entirely through MTM and booked at close."""
        self.positions[pos.position_id] = pos

    def close_position(
        self,
        position_id: str,
        close_price: float,
        ts: pd.Timestamp,
        reason: str = "normal",
    ) -> float:
        """Close a position; return realized PnL (dollars)."""
        pos = self.positions.pop(position_id, None)
        if pos is None:
            return 0.0
        pnl = (close_price - pos.open_price) * pos.quantity * CONTRACT_UNIT
        self.cash += pnl
        self._log_close(pos, close_price, ts, pnl, reason)
        return pnl

    def close_all(
        self,
        ts: pd.Timestamp,
        universe,
        surface,
        pricer,
        reason: str = "stop_loss",
    ) -> float:
        """Close every open position at current market prices."""
        ids = list(self.positions.keys())
        total_pnl = 0.0
        for pid in ids:
            pos = self.positions[pid]
            close_px = pos.current_price(ts, universe, surface, pricer)
            total_pnl += self.close_position(pid, close_px, ts, reason)
        # Unwind hedge
        if self.hedge_shares != 0.0:
            spot = universe.spot(ts)
            self.cash += self.hedge_shares * spot
            self.hedge_shares = 0.0
        return total_pnl

    # ------------------------------------------------------------------
    # Hedge management
    # ------------------------------------------------------------------

    def update_hedge(self, target_shares: float, price: float) -> float:
        """Trade to reach target_shares; return shares traded (signed)."""
        delta_shares = target_shares - self.hedge_shares
        self.cash -= delta_shares * price   # buy positive = spend cash
        self.hedge_shares = target_shares
        return delta_shares

    # ------------------------------------------------------------------
    # MTM and Greeks
    # ------------------------------------------------------------------

    def mark_to_market(self, ts: pd.Timestamp, universe, surface, pricer) -> float:
        """Compute total portfolio equity: cash + option MTM + hedge value."""
        spot = universe.spot(ts)
        option_mtm = 0.0
        if self.positions:
            flags, S_arr, K_arr, T_arr, r_arr, sig_arr = [], [], [], [], [], []
            pos_list = list(self.positions.values())
            for pos in pos_list:
                T = pos.time_to_expiry(ts)
                sig = surface.iv_for_strike(ts, pos.K, T, pos.flag)
                flags.append(pos.flag)
                S_arr.append(universe.spot(ts))
                K_arr.append(pos.K)
                T_arr.append(T)
                r_arr.append(universe.rate(ts))
                sig_arr.append(sig)
            prices, _ = pricer.batch_price_and_delta(
                np.asarray(flags),
                np.asarray(S_arr),
                np.asarray(K_arr),
                np.asarray(T_arr),
                np.asarray(r_arr),
                np.asarray(sig_arr),
            )
            for i, pos in enumerate(pos_list):
                option_mtm += (prices[i] - pos.open_price) * pos.quantity * CONTRACT_UNIT
        return self.cash + option_mtm + self.hedge_shares * spot

    def net_delta(self, ts: pd.Timestamp, universe, surface, pricer) -> float:
        """Portfolio net delta in share-equivalents (includes hedge)."""
        if not self.positions:
            return self.hedge_shares
        flags, S_arr, K_arr, T_arr, r_arr, sig_arr = [], [], [], [], [], []
        pos_list = list(self.positions.values())
        qtys = []
        for pos in pos_list:
            T = pos.time_to_expiry(ts)
            sig = surface.iv_for_strike(ts, pos.K, T, pos.flag)
            flags.append(pos.flag)
            S_arr.append(universe.spot(ts))
            K_arr.append(pos.K)
            T_arr.append(T)
            r_arr.append(universe.rate(ts))
            sig_arr.append(sig)
            qtys.append(pos.quantity)
        _, deltas = pricer.batch_price_and_delta(
            np.asarray(flags),
            np.asarray(S_arr),
            np.asarray(K_arr),
            np.asarray(T_arr),
            np.asarray(r_arr),
            np.asarray(sig_arr),
        )
        option_delta = float(np.dot(np.asarray(deltas), np.asarray(qtys)) * CONTRACT_UNIT)
        return option_delta + self.hedge_shares

    def position_deltas(self, ts: pd.Timestamp, universe, surface, pricer) -> dict:
        """Per-position delta dict (position_id → delta in share-eq)."""
        if not self.positions:
            return {}
        pos_list = list(self.positions.values())
        flags, S_arr, K_arr, T_arr, r_arr, sig_arr, qtys = [], [], [], [], [], [], []
        for pos in pos_list:
            T = pos.time_to_expiry(ts)
            sig = surface.iv_for_strike(ts, pos.K, T, pos.flag)
            flags.append(pos.flag)
            S_arr.append(universe.spot(ts))
            K_arr.append(pos.K)
            T_arr.append(T)
            r_arr.append(universe.rate(ts))
            sig_arr.append(sig)
            qtys.append(pos.quantity)
        _, deltas = pricer.batch_price_and_delta(
            np.asarray(flags),
            np.asarray(S_arr),
            np.asarray(K_arr),
            np.asarray(T_arr),
            np.asarray(r_arr),
            np.asarray(sig_arr),
        )
        return {
            pos.position_id: float(deltas[i]) * pos.quantity * CONTRACT_UNIT
            for i, pos in enumerate(pos_list)
        }

    # ------------------------------------------------------------------
    # Trade log (for ML feature building)
    # ------------------------------------------------------------------

    def log_daily_pnl(self, ts: pd.Timestamp, universe, surface, pricer) -> None:
        """Append a per-position daily MTM PnL row to _trade_log."""
        if not self.positions:
            return
        pos_list = list(self.positions.values())
        flags, S_arr, K_arr, T_arr, r_arr, sig_arr = [], [], [], [], [], []
        for pos in pos_list:
            T = pos.time_to_expiry(ts)
            sig = surface.iv_for_strike(ts, pos.K, T, pos.flag)
            flags.append(pos.flag)
            S_arr.append(universe.spot(ts))
            K_arr.append(pos.K)
            T_arr.append(T)
            r_arr.append(universe.rate(ts))
            sig_arr.append(sig)
        prices, _ = pricer.batch_price_and_delta(
            np.asarray(flags),
            np.asarray(S_arr),
            np.asarray(K_arr),
            np.asarray(T_arr),
            np.asarray(r_arr),
            np.asarray(sig_arr),
        )
        for i, pos in enumerate(pos_list):
            pnl = (prices[i] - pos.open_price) * pos.quantity * CONTRACT_UNIT
            self._trade_log.append({
                "ts": ts,
                "position_id": pos.position_id,
                "flag": pos.flag,
                "K": pos.K,
                "open_date": pos.open_date,
                "expiry_date": pos.expiry_date,
                "daily_pnl": pnl,
            })

    @property
    def trade_log(self) -> pd.DataFrame:
        if not self._trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self._trade_log)

    def _log_close(self, pos: OptionPosition, close_price: float, ts, pnl: float, reason: str) -> None:
        self._trade_log.append({
            "ts": ts,
            "position_id": pos.position_id,
            "flag": pos.flag,
            "K": pos.K,
            "open_date": pos.open_date,
            "expiry_date": pos.expiry_date,
            "daily_pnl": pnl,
            "close_reason": reason,
        })

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Portfolio(cash={self.cash:.2f}, positions={len(self.positions)}, "
            f"hedge_shares={self.hedge_shares:.1f})"
        )
