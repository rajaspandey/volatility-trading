"""BaseStrategy: abstract interface for all strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all volatility strategies.

    The backtester calls on_open() at 10AM every trading day.
    Strategies return new OptionPosition objects to open.
    The backtester handles expiry automatically via ExpiryCheckEvent.
    """

    @abstractmethod
    def on_open(
        self,
        ts: pd.Timestamp,
        portfolio,
        universe,
        surface,
        pricer,
    ) -> List:
        """Called at 10AM each trading day. Return list of OptionPosition to open."""
        ...

    def on_risk_exit(self, ts: pd.Timestamp) -> None:
        """Called when a risk event (stop-loss / delta-exit) fires."""
        pass
