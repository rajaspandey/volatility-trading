"""Run Backtest page — configure and execute a strategy live."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.set_page_config(page_title="Run Backtest", page_icon="▶️", layout="wide")
st.title("Run a Backtest")


@st.cache_data(ttl=3600)
def _load_data():
    daily  = pd.read_parquet("data/processed/market_daily.parquet")
    hourly = pd.read_parquet("data/processed/market_hourly.parquet")
    return daily, hourly


@st.cache_data(ttl=3600)
def _load_config():
    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open("config/strategy_params.yaml") as f:
        params = yaml.safe_load(f)
    return settings, params


settings, params = _load_config()
cap = settings["portfolio"]["initial_capital"]

# ── Sidebar config ───────────────────────────────────────────────────────────
st.sidebar.header("Strategy Config")

strategy_choice = st.sidebar.selectbox(
    "Strategy",
    ["Short Straddle", "Strangle (ss=0%)", "Strangle (ss=5%)", "Strangle (ss=10%)", "Skew Arb"],
)

start_date = st.sidebar.date_input(
    "Start Date", value=pd.Timestamp("2022-01-03").date(),
    min_value=pd.Timestamp("2020-01-01").date(),
)
end_date = st.sidebar.date_input(
    "End Date", value=pd.Timestamp("2024-12-31").date(),
    max_value=pd.Timestamp.today().date(),
)

stop_loss = st.sidebar.slider("Stop-Loss (%)", 1, 10, 2) / 100.0
delta_exit = st.sidebar.slider("Delta-Exit Threshold", 0.25, 2.0, 0.75, 0.05)

# ── Run ──────────────────────────────────────────────────────────────────────
if st.sidebar.button("▶  Run Backtest", type="primary"):
    daily, hourly = _load_data()

    from src.data.universe import MarketUniverse
    from src.engine.backtester import Backtester
    from src.reporting.metrics import compute_metrics
    from src.reporting.plots import equity_curves, drawdown_chart, monthly_returns_heatmap

    universe = MarketUniverse(daily, hourly)

    # Patch settings with sidebar values
    run_settings = {**settings}
    run_settings["risk"] = {
        "stop_loss_pct": stop_loss,
        "delta_exit_threshold": delta_exit,
    }

    bt = Backtester(universe, run_settings)

    strategy_map = {
        "Short Straddle":     ("straddle",  None),
        "Strangle (ss=0%)":   ("strangle",  0.0),
        "Strangle (ss=5%)":   ("strangle",  0.05),
        "Strangle (ss=10%)":  ("strangle",  0.1),
        "Skew Arb":           ("skew_arb",  None),
    }
    strat_type, ss_val = strategy_map[strategy_choice]

    effective_start = (
        settings["data"]["start_date"] if strat_type == "skew_arb" else str(start_date)
    )

    with st.spinner(f"Running {strategy_choice}…"):
        if strat_type == "straddle":
            from src.strategies.straddle import ShortStraddleStrategy
            strat = ShortStraddleStrategy()
        elif strat_type == "strangle":
            from src.strategies.strangle import WeeklyStrangleStrategy
            strat = WeeklyStrangleStrategy(ss_val)
        else:
            from src.strategies.skew_arb import WeightedSkewArbStrategy
            strat = WeightedSkewArbStrategy()

        res = bt.run(strat, start=effective_start, end=str(end_date))

    eq = res["equity_curve"]
    if strat_type == "skew_arb":
        eq = eq[eq.index >= str(start_date)]

    if eq.empty:
        st.error("No equity data returned. Check the date range and data availability.")
        st.stop()

    m = compute_metrics(eq, cap)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ann. Return",  f"{m['ann_return']:+.2%}")
    c2.metric("Sharpe",       f"{m['sharpe']:.2f}")
    c3.metric("Max Drawdown", f"{m['max_drawdown']:.2%}")
    c4.metric("Ann. Vol",     f"{m['ann_vol']:.2%}")
    c5.metric("Win Rate",     f"{m['win_rate']:.2%}")

    risk_df = res.get("risk_events", pd.DataFrame())
    if not risk_df.empty:
        st.warning(f"{len(risk_df)} risk event(s) fired during backtest.")

    # Charts
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.plotly_chart(
            equity_curves({strategy_choice: eq}, f"{strategy_choice} Equity"),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            drawdown_chart(eq["equity"], "Drawdown"),
            use_container_width=True,
        )

    st.plotly_chart(
        monthly_returns_heatmap(eq["equity"], "Monthly Returns"),
        use_container_width=True,
    )

    # Trade log preview
    tlog = res.get("trade_log", pd.DataFrame())
    if not tlog.empty:
        st.subheader("Trade Log (last 20 entries)")
        st.dataframe(tlog.tail(20), use_container_width=True, hide_index=True)
else:
    st.info("Configure the strategy in the sidebar and click **Run Backtest**.")
