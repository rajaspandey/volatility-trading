"""DataLoader: fetches and caches all raw market data."""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Tickers to download
DAILY_TICKERS = ["SPY", "^GSPC", "^VIX", "^VIX9D", "^SKEW", "^IRX"]
HOURLY_TICKER = "SPY"


def _parquet_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.parquet"


def _is_stale(path: Path, max_age_hours: float = 6.0) -> bool:
    if not path.exists():
        return True
    age = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).total_seconds()
    return age > max_age_hours * 3600


def fetch_daily(
    tickers: list[str],
    start: str,
    end: str | None,
    cache_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Download daily OHLCV for all tickers; cache as parquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _parquet_path(cache_dir, "daily_raw")

    if not force and not _is_stale(path):
        logger.info("Loading daily data from cache: %s", path)
        return pd.read_parquet(path)

    logger.info("Downloading daily data for %s from %s …", tickers, start)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,  # avoid sqlite lock on macOS
        )
    # Flatten MultiIndex columns → (field, ticker) tuples already; keep Close
    # yfinance returns MultiIndex (field, ticker) when multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].copy()
        close.index = pd.to_datetime(close.index)
        close.index.name = "date"
    else:
        close = df[["Close"]].copy()
        close.columns = [tickers[0]]
        close.index = pd.to_datetime(close.index)
        close.index.name = "date"

    # Rename to clean names
    rename = {
        "SPY": "spy",
        "^GSPC": "spx",
        "^VIX": "vix",
        "^VIX9D": "vix9d",
        "^SKEW": "skew",
        "^IRX": "irx",
    }
    close.rename(columns={k: v for k, v in rename.items() if k in close.columns}, inplace=True)
    close.dropna(how="all", inplace=True)
    close.to_parquet(path)
    logger.info("Saved daily data: %s rows → %s", len(close), path)
    return close


def fetch_hourly(
    ticker: str,
    start: str,
    end: str | None,
    cache_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Download 1H OHLCV for SPY; cache as parquet.

    yfinance only supports 1H for the last ~730 days. We cap start to
    max(start, today - 729 days) to avoid an API error.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _parquet_path(cache_dir, "hourly_raw")

    if not force and not _is_stale(path):
        logger.info("Loading hourly data from cache: %s", path)
        return pd.read_parquet(path)

    # yfinance 1H limit: ~2 years
    earliest_1h = (pd.Timestamp.today() - pd.Timedelta(days=728)).strftime("%Y-%m-%d")
    effective_start = max(start, earliest_1h)

    logger.info("Downloading 1H data for %s from %s …", ticker, effective_start)
    df = yf.download(
        ticker,
        start=effective_start,
        end=end,
        interval="1h",
        auto_adjust=True,
        progress=False,
    )
    df.index = pd.to_datetime(df.index)
    # Localize to US/Eastern for 10AM/2PM identification
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")
    df.index.name = "ts"
    # Flatten MultiIndex columns (yfinance ≥0.2 returns (field, ticker))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f[0].lower() for f in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.to_parquet(path)
    logger.info("Saved hourly data: %s rows → %s", len(df), path)
    return df


def fetch_fred_rate(series: str, start: str, end: str | None, cache_dir: Path, force: bool = False) -> pd.Series:
    """Download risk-free rate from FRED (TB3MS or similar)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _parquet_path(cache_dir, f"fred_{series.lower()}")

    if not force and not _is_stale(path, max_age_hours=24):
        return pd.read_parquet(path)["rate"]

    try:
        from pandas_datareader import data as web  # type: ignore

        logger.info("Downloading FRED %s …", series)
        raw = web.get_data_fred(series, start=start, end=end)
        s = raw[series].rename("rate") / 100.0  # percent → decimal
        s.index = pd.to_datetime(s.index)
        s.index.name = "date"
        s.to_frame().to_parquet(path)
        return s
    except Exception as exc:
        logger.warning("FRED fetch failed (%s); falling back to IRX from daily data", exc)
        return pd.Series(dtype=float, name="rate")


class DataLoader:
    """Orchestrates all raw data fetching and caching."""

    def __init__(self, settings: dict):
        self.settings = settings
        self.cache_dir = Path(settings["data"]["cache_dir"])
        self.start = settings["data"]["start_date"]
        self.end = settings["data"].get("end_date")  # None = today

    def fetch_all(self, force: bool = False) -> None:
        """Download and cache all raw data."""
        tickers = list(self.settings["data"]["tickers"].values())
        fetch_daily(tickers, self.start, self.end, self.cache_dir, force=force)
        fetch_hourly("SPY", self.start, self.end, self.cache_dir, force=force)
        fred_series = self.settings["data"].get("fred_series", "TB3MS")
        fetch_fred_rate(fred_series, self.start, self.end, self.cache_dir, force=force)
        logger.info("All raw data fetched.")

    def load_daily_raw(self) -> pd.DataFrame:
        return pd.read_parquet(_parquet_path(self.cache_dir, "daily_raw"))

    def load_hourly_raw(self) -> pd.DataFrame:
        return pd.read_parquet(_parquet_path(self.cache_dir, "hourly_raw"))

    def load_fred_rate(self) -> pd.Series:
        fred_series = self.settings["data"].get("fred_series", "TB3MS")
        path = _parquet_path(self.cache_dir, f"fred_{fred_series.lower()}")
        if path.exists():
            return pd.read_parquet(path)["rate"]
        return pd.Series(dtype=float, name="rate")
