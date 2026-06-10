"""Volatility Trading Strategies Dashboard.

Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Volatility Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Cached data loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_settings():
    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open("config/strategy_params.yaml") as f:
        params = yaml.safe_load(f)
    return settings, params


@st.cache_data(ttl=3600)
def load_market_data():
    daily  = pd.read_parquet("data/processed/market_daily.parquet")
    hourly = pd.read_parquet("data/processed/market_hourly.parquet")
    return daily, hourly


@st.cache_resource
def get_universe_and_surface():
    from src.data.universe import MarketUniverse
    from src.options.iv_surface import IVSurface
    settings, params = load_settings()
    daily, hourly = load_market_data()
    universe = MarketUniverse(daily, hourly)
    surface  = IVSurface(universe, skew_scale=params["iv_surface"]["skew_scale"])
    return universe, surface


@st.cache_data(ttl=600)
def run_backtest(strategy_name: str, start: str, end: str, params_key: str = "") -> dict:
    """Run a backtest and cache the result."""
    settings, params = load_settings()
    daily, hourly = load_market_data()

    from src.data.universe import MarketUniverse
    from src.engine.backtester import Backtester
    universe = MarketUniverse(daily, hourly)
    bt = Backtester(universe, settings)

    if strategy_name == "straddle":
        from src.strategies.straddle import ShortStraddleStrategy
        strategy = ShortStraddleStrategy()
    elif strategy_name.startswith("strangle_"):
        ss = float(strategy_name.split("_")[1])
        from src.strategies.strangle import WeeklyStrangleStrategy
        strategy = WeeklyStrangleStrategy(ss)
    elif strategy_name == "skew_arb":
        from src.strategies.skew_arb import WeightedSkewArbStrategy
        strategy = WeightedSkewArbStrategy(
            lambda_lookback=params["skew_arb"]["lambda_lookback_days"]
        )
    else:
        st.error(f"Unknown strategy: {strategy_name}")
        return {}

    return bt.run(strategy, start=start, end=end)


@st.cache_data(ttl=600)
def compute_metrics_cached(equity_json: str, initial_capital: float) -> dict:
    from src.reporting.metrics import compute_metrics
    eq = pd.read_json(equity_json, orient="split")
    return compute_metrics(eq, initial_capital)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_LABELS = {
    "straddle":     "Short Straddle",
    "strangle_0.0": "Strangle (ATM)",
    "strangle_0.05":"Strangle (ss=5%)",
    "strangle_0.1": "Strangle (ss=10%)",
    "skew_arb":     "Weighted Skew Arb",
}
PAPER_TARGETS = {
    "straddle":    {"return": -0.033, "sharpe": -0.58},
    "strangle_0.1":{"return":  0.029, "sharpe":  0.79},
    "skew_arb":    {"return":  0.115, "sharpe":  0.93},
}
COLORS = {
    "straddle":     "#636EFA",
    "strangle_0.0": "#EF553B",
    "strangle_0.05":"#00CC96",
    "strangle_0.1": "#AB63FA",
    "skew_arb":     "#FFA15A",
}


def equity_curve_fig(results: dict[str, pd.DataFrame], title: str = "Equity Curves") -> go.Figure:
    fig = go.Figure()
    for key, eq in results.items():
        if eq is None or eq.empty:
            continue
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq["equity"],
            mode="lines", name=STRATEGY_LABELS.get(key, key),
            line=dict(color=COLORS.get(key, None), width=2),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def metrics_table(metrics_dict: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, m in metrics_dict.items():
        label = STRATEGY_LABELS.get(key, key)
        target = PAPER_TARGETS.get(key, {})
        rows.append({
            "Strategy": label,
            "Ann Return": f"{m['ann_return']:+.2%}",
            "Sharpe": f"{m['sharpe']:.2f}",
            "Max DD": f"{m['max_drawdown']:.2%}",
            "Ann Vol": f"{m['ann_vol']:.2%}",
            "Win Rate": f"{m['win_rate']:.2%}",
            "Paper Return": f"{target['return']:+.2%}" if target else "—",
            "Paper Sharpe": f"{target['sharpe']:.2f}" if target else "—",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Volatility Strategies")
st.sidebar.markdown("*Based on Duke University research*")

settings, params = load_settings()
cap = settings["portfolio"]["initial_capital"]

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Strategy Analysis", "Market Data", "IV Surface", "About"],
    index=0,
)

# Date range selectors
st.sidebar.markdown("---")
st.sidebar.subheader("Backtest Period")
start_date = st.sidebar.date_input(
    "Start Date", value=pd.Timestamp("2022-01-03").date(),
    min_value=pd.Timestamp("2020-01-01").date(),
)
end_date = st.sidebar.date_input(
    "End Date", value=pd.Timestamp("2022-11-30").date(),
    max_value=pd.Timestamp.today().date(),
)

# ─────────────────────────────────────────────────────────────────────────────
# Main pages
# ─────────────────────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Volatility Trading Strategy Replication")
    st.markdown("""
    This dashboard replicates three volatility-selling strategies from the Duke University
    working paper *"Trading Volatility Using Options on SPX"* using freely available market
    data (VIX, VIX9D, CBOE SKEW) and synthetic Black-Scholes option pricing.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Capital", f"${cap:,.0f}")
    col2.metric("Underlying", "SPY (primary)")
    col3.metric("Data Source", "yfinance + FRED")

    st.markdown("---")

    # Load cached equity curves if available
    eq_dir = Path("data/processed")
    available = {}
    strategy_keys = ["straddle", "strangle_ss0", "strangle_ss5", "strangle_ss10", "skew_arb"]
    key_map = {
        "straddle": "straddle", "strangle_ss0": "strangle_0.0",
        "strangle_ss5": "strangle_0.05", "strangle_ss10": "strangle_0.1",
        "skew_arb": "skew_arb",
    }
    for k, label_key in key_map.items():
        fp = eq_dir / f"equity_{k}.parquet"
        if fp.exists():
            eq = pd.read_parquet(fp)
            available[label_key] = eq

    if available:
        st.subheader("Equity Curves (Last Backtest Run)")
        fig = equity_curve_fig(available, "All Strategies — Jan-Nov 2022")
        st.plotly_chart(fig, use_container_width=True)

        from src.reporting.metrics import compute_metrics
        metrics = {k: compute_metrics(eq, cap) for k, eq in available.items()}
        st.dataframe(metrics_table(metrics), use_container_width=True, hide_index=True)

        st.caption(
            "**Note**: Returns differ from the paper's OS-A benchmark due to synthetic option pricing "
            "(VIX-derived IV vs. real options), daily close-to-close delta hedging, and no transaction costs."
        )
    else:
        st.info("No backtest results found. Run `python scripts/run_all.py` to generate equity curves.")

    # Paper comparison table
    st.markdown("---")
    st.subheader("Paper OS-A Benchmark (Jan–Nov 2022)")
    paper_df = pd.DataFrame([
        {"Strategy": "Short Straddle", "Paper Return": "-3.3%", "Paper Sharpe": "-0.58"},
        {"Strategy": "Short Strangle (ss=10%)", "Paper Return": "+2.94%", "Paper Sharpe": "0.79"},
        {"Strategy": "Weighted Skew Arb", "Paper Return": "+11.5%", "Paper Sharpe": "0.93"},
        {"Strategy": "ML Strangle S₂ (ss=10%)", "Paper Return": "+3.4%", "Paper Sharpe": "1.06"},
    ])
    st.dataframe(paper_df, use_container_width=True, hide_index=True)


elif page == "Strategy Analysis":
    st.title("Strategy Analysis")

    selected_strategy = st.selectbox(
        "Select Strategy",
        list(STRATEGY_LABELS.keys()),
        format_func=lambda k: STRATEGY_LABELS[k],
    )

    # For skew arb, use a longer start to allow warm-up
    run_start = settings["data"]["start_date"] if selected_strategy == "skew_arb" else str(start_date)
    run_end = str(end_date)

    if st.button("Run Backtest", type="primary"):
        with st.spinner(f"Running {STRATEGY_LABELS[selected_strategy]} backtest…"):
            strategy_key = selected_strategy.replace("0.0", "0.0").replace("0.05", "0.05")
            # Parse strategy key for strangle
            if selected_strategy.startswith("strangle_"):
                ss_val = float(selected_strategy.split("_")[1])
                res = run_backtest(f"strangle_{ss_val}", run_start, run_end)
            else:
                res = run_backtest(selected_strategy, run_start, run_end)

        eq = res.get("equity_curve", pd.DataFrame())
        if eq.empty:
            st.warning("No equity data returned.")
        else:
            if selected_strategy == "skew_arb":
                eq_display = eq[eq.index >= str(start_date)]
            else:
                eq_display = eq

            from src.reporting.metrics import compute_metrics
            m = compute_metrics(eq_display, cap)

            # KPI row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ann. Return", f"{m['ann_return']:+.2%}")
            c2.metric("Sharpe Ratio", f"{m['sharpe']:.2f}")
            c3.metric("Max Drawdown", f"{m['max_drawdown']:.2%}")
            c4.metric("Win Rate", f"{m['win_rate']:.2%}")

            # Equity curve
            fig = equity_curve_fig(
                {selected_strategy: eq_display},
                f"{STRATEGY_LABELS[selected_strategy]} — Equity Curve",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Drawdown chart
            rolling_max = eq_display["equity"].cummax()
            dd = (eq_display["equity"] - rolling_max) / rolling_max
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values * 100,
                fill="tozeroy", mode="lines",
                line=dict(color="#EF553B", width=1),
                name="Drawdown %",
            ))
            fig_dd.update_layout(
                title="Drawdown", xaxis_title="Date", yaxis_title="%",
                height=250, margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_dd, use_container_width=True)

            # Risk events
            risk_df = res.get("risk_events", pd.DataFrame())
            if not risk_df.empty:
                st.subheader("Risk Events")
                st.dataframe(risk_df, use_container_width=True, hide_index=True)
            else:
                st.info("No risk events (stop-loss / delta-exit) triggered.")


elif page == "Market Data":
    st.title("Market Data")

    daily, hourly = load_market_data()

    tab1, tab2 = st.tabs(["Daily Signals", "Hourly SPY"])

    with tab1:
        col_select = st.multiselect(
            "Select columns",
            ["spy", "vix", "vix9d", "skew", "iv_30", "rv_30", "iv_rv_diff_30", "iv_rank_30"],
            default=["spy", "vix", "iv_rv_diff_30"],
        )
        if col_select:
            fig = go.Figure()
            for col in col_select:
                if col in daily.columns:
                    y = daily[col]
                    fig.add_trace(go.Scatter(
                        x=daily.index, y=y,
                        mode="lines", name=col,
                        yaxis="y" if col in ("spy",) else "y2",
                    ))
            fig.update_layout(
                title="Daily Market Signals",
                height=400,
                yaxis=dict(title="SPY Price ($)"),
                yaxis2=dict(title="Indices / Vol", overlaying="y", side="right"),
                hovermode="x unified",
                margin=dict(l=40, r=60, t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Data")
        st.dataframe(daily.tail(10).round(4), use_container_width=True)

    with tab2:
        if not hourly.empty:
            recent = hourly.tail(1000)
            fig = go.Figure(data=[go.Candlestick(
                x=recent.index,
                open=recent["open"], high=recent["high"],
                low=recent["low"], close=recent["close"],
                name="SPY 1H",
            )])
            fig.update_layout(
                title="SPY 1-Hour OHLCV (last 1000 bars)",
                xaxis_title="Timestamp (ET)",
                yaxis_title="Price ($)",
                height=420,
                xaxis_rangeslider_visible=False,
                margin=dict(l=40, r=20, t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hourly data available.")


elif page == "IV Surface":
    st.title("Implied Volatility Surface")

    universe, surface = get_universe_and_surface()

    # Date selector
    daily, _ = load_market_data()
    available_dates = daily.index.tolist()
    selected_date = st.select_slider(
        "Select Date",
        options=available_dates,
        value=available_dates[-1],
        format_func=lambda d: str(d.date()),
    )
    ts = pd.Timestamp(selected_date)

    S = universe.spot(ts)
    r = universe.rate(ts)
    vix = universe.iv_atm_30d(ts)
    skew_idx = universe.skew_index(ts)

    col1, col2, col3 = st.columns(3)
    col1.metric("SPY Spot", f"${S:.2f}")
    col2.metric("VIX (ATM 30d IV)", f"{vix:.1%}")
    col3.metric("CBOE SKEW", f"{skew_idx:.1f}")

    # Compute IV smile across strikes
    tenors_days = [5, 9, 14, 21, 30]
    strike_pcts = np.arange(0.85, 1.16, 0.01)
    strikes = strike_pcts * S

    fig = go.Figure()
    for T_days in tenors_days:
        T = T_days / 252.0
        ivs_put  = [surface.iv_for_strike(ts, K, T, "p") for K in strikes]
        ivs_call = [surface.iv_for_strike(ts, K, T, "c") for K in strikes]
        # Show put IVs for K < S, call IVs for K > S (standard convention)
        ivs = [iv_p if K <= S else iv_c for K, iv_p, iv_c in zip(strikes, ivs_put, ivs_call)]
        fig.add_trace(go.Scatter(
            x=strike_pcts * 100, y=[iv * 100 for iv in ivs],
            mode="lines", name=f"{T_days}d",
        ))
    fig.add_vline(x=100.0, line_dash="dash", line_color="gray", annotation_text="ATM")
    fig.update_layout(
        title=f"IV Smile — {ts.date()}",
        xaxis_title="Strike (% of Spot)",
        yaxis_title="Implied Vol (%)",
        hovermode="x unified",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # IV vs RV over time
    st.subheader("IV − RV Spread (30-day)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=daily.index, y=daily["iv_30"] * 100,
        mode="lines", name="VIX (IV 30d)", line=dict(color="#636EFA"),
    ))
    fig2.add_trace(go.Scatter(
        x=daily.index, y=daily["rv_30"] * 100,
        mode="lines", name="RV 30d", line=dict(color="#EF553B"),
    ))
    if "iv_rv_diff_30" in daily.columns:
        fig2.add_trace(go.Scatter(
            x=daily.index, y=daily["iv_rv_diff_30"] * 100,
            mode="lines", name="IV − RV", line=dict(color="#00CC96", dash="dot"),
        ))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(
        title="Implied vs Realized Volatility",
        xaxis_title="Date", yaxis_title="Vol (%)",
        hovermode="x unified", height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)


elif page == "About":
    st.title("About This Project")
    st.markdown("""
    ## Volatility Trading Strategy Replication

    This project replicates three volatility-selling strategies from the Duke University
    working paper *"Trading Volatility Using Options on the S&P 500"* using freely available
    market data and a custom Python backtesting framework.

    ### Strategies Implemented

    | # | Strategy | Description |
    |---|----------|-------------|
    | 1 | **Short Straddle** | Monthly ATM straddle (sell 1 call + 1 put), delta-hedged twice daily |
    | 2 | **Short Strangle** | Daily 5-day strangle (3 widths: 0%, 5%, 10% × σ), rolling portfolio |
    | 3 | **Weighted Skew Arb** | Monthly long ATM put + short λ OTM puts, 2× leverage |
    | 4 | **ML Strangle (S₁/S₂)** | RF-gated strangle using 43-feature daily signal |

    ### Data Sources
    - **SPY / ^VIX / ^VIX9D / ^SKEW / ^IRX**: yfinance (free)
    - **3-month T-bill rate**: FRED via pandas-datareader (free)
    - **Options**: *Synthetic* — Black-Scholes with VIX-derived ATM IV + SKEW-adjusted smile

    ### Key Assumptions
    - Option prices synthesized using VIX (30-day ATM IV) and VIX9D (9-day ATM IV)
    - OTM skew via CBOE SKEW index: `IV(K) = ATM_IV × (1 + α × log(K/F) / (ATM_IV × √T))`
    - Delta hedge executed once per day at close (vs. 2× intraday in the paper)
    - No transaction costs or bid-ask spreads

    ### Why Results Differ from the Paper
    The paper used **real SPX options chain data** (proprietary). Our synthetic approach
    causes systematic differences:
    - Gamma losses are understated (daily hedging misses intraday moves)
    - No transaction costs overstates returns by ~1-2%/year
    - VIX-derived IV may differ from actual market prices

    ### Tech Stack
    - **Python 3.13** with numpy, scipy, pandas, scikit-learn
    - **Streamlit** dashboard
    - **Black-Scholes** via custom scipy implementation (numba-free)

    ---
    *Rajas Pandey · pandey.rajas629@gmail.com*
    """)
