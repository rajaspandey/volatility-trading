"""Event dataclasses for the backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Event:
    ts: pd.Timestamp


@dataclass
class OpenPositionEvent(Event):
    """Fired at 10AM: strategies may open new positions."""
    pass


@dataclass
class HedgeEvent(Event):
    """Fired at 10AM and 2PM: execute delta hedge."""
    session: str = "am"  # "am" or "pm"


@dataclass
class RiskCheckEvent(Event):
    """Fired after each hedge: run stop-loss and delta-exit checks."""
    session: str = "am"


@dataclass
class ExpiryCheckEvent(Event):
    """Fired EOD: close any positions expiring on or before this date."""
    pass


@dataclass
class EODEvent(Event):
    """Fired EOD: record equity curve point and daily PnL."""
    pass
