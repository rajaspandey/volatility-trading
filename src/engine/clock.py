"""TradingClock: generates ordered event list from hourly market data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import pandas as pd

from src.engine.events import (
    Event,
    OpenPositionEvent,
    HedgeEvent,
    RiskCheckEvent,
    ExpiryCheckEvent,
    EODEvent,
)


@dataclass
class TradingClock:
    """Generates the event sequence for the backtester.

    Per-day event order:
        10AM bar  → OpenPositionEvent → HedgeEvent(am) → RiskCheckEvent
        2PM bar   → HedgeEvent(pm) → RiskCheckEvent
        EOD       → ExpiryCheckEvent → EODEvent

    yfinance 1H bars in US/Eastern:
        9:30 bar (hour=9)  → morning open proxy
        13:30 bar (hour=13) → afternoon proxy

    Only emits events on trading days present in both daily and hourly data.
    """

    hourly: pd.DataFrame   # index: tz-aware US/Eastern timestamps
    daily: pd.DataFrame    # index: tz-naive dates (DatetimeIndex)

    def events(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> Iterator[Event]:
        """Yield events in chronological order between start and end."""
        start_ts = pd.Timestamp(start)
        end_ts   = pd.Timestamp(end)

        # Trading days = dates present in daily data within range
        trading_dates = self.daily.index[
            (self.daily.index >= start_ts) & (self.daily.index <= end_ts)
        ]

        for dt in trading_dates:
            date_str = dt.date()

            # Get 10AM and 2PM bars for this date
            am_bar = self._get_bar(dt, hour=9)   # 9:30 open bar
            pm_bar = self._get_bar(dt, hour=13)  # 13:30 bar

            eod_ts = pd.Timestamp(date_str)

            if am_bar is None and pm_bar is None:
                # No intraday data: use daily close as single pricing point
                yield OpenPositionEvent(ts=eod_ts)
                yield HedgeEvent(ts=eod_ts, session="am")
                yield RiskCheckEvent(ts=eod_ts, session="am")
                yield ExpiryCheckEvent(ts=eod_ts)
                yield EODEvent(ts=eod_ts)
                continue

            if am_bar is not None:
                yield OpenPositionEvent(ts=am_bar)
                yield HedgeEvent(ts=am_bar, session="am")
                yield RiskCheckEvent(ts=am_bar, session="am")
            else:
                # Hourly missing for AM: fall back to daily open
                yield OpenPositionEvent(ts=eod_ts)
                yield HedgeEvent(ts=eod_ts, session="am")
                yield RiskCheckEvent(ts=eod_ts, session="am")

            if pm_bar is not None:
                yield HedgeEvent(ts=pm_bar, session="pm")
                yield RiskCheckEvent(ts=pm_bar, session="pm")

            yield ExpiryCheckEvent(ts=eod_ts)
            yield EODEvent(ts=eod_ts)

    def _get_bar(self, dt: pd.Timestamp, hour: int) -> pd.Timestamp | None:
        """Return the hourly bar timestamp for dt at the given hour, or None.

        Always returns a tz-naive timestamp so all event timestamps are uniform.
        """
        dt_date = dt.date()
        mask = (
            (self.hourly.index.date == dt_date) &
            (self.hourly.index.hour == hour)
        )
        rows = self.hourly.index[mask]
        if len(rows) == 0:
            return None
        ts = rows[0]
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
