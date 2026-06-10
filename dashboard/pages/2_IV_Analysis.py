"""IV Analysis page — IV surface, smile, IV-RV spread."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.set_page_config(page_title="IV Analysis", page_icon="🌊", layout="wide")
st.title("Implied Volatility Analysis")


@st.cache_data(ttl=3600)
def _load():
    daily  = pd.read_parquet("data/processed/market_daily.parquet")
    hourly = pd.read_parquet("data/processed/market_hourly.parquet")
    return daily, hourly


@st.cache_resource
def _universe_surface():
    import yaml
    from src.data.universe import MarketUniverse
    from src.options.iv_surface import IVSurface
    daily, hourly = _load()
    with open("config/strategy_params.yaml") as f:
        import yaml
        params = yaml.safe_load(f)
    u = MarketUniverse(daily, hourly)
    s = IVSurface(u, skew_scale=params["iv_surface"]["skew_scale"])
    return u, s


daily, hourly = _load()
universe, surface = _universe_surface()

from src.reporting.plots import iv_rv_spread, iv_smile

tab1, tab2 = st.tabs(["IV−RV Spread", "IV Smile"])

with tab1:
    tenor = st.radio("Tenor", ["30", "5"], horizontal=True)
    fig = iv_rv_spread(daily, tenor=tenor)
    st.plotly_chart(fig, use_container_width=True)

    # IV Rank
    rank_col = f"iv_rank_{tenor}"
    if rank_col in daily.columns:
        import plotly.graph_objects as go
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=daily.index, y=daily[rank_col] * 100,
            mode="lines", name=f"IV Rank {tenor}d",
            line=dict(color="#AB63FA"),
        ))
        fig2.add_hline(y=50, line_dash="dash", line_color="gray")
        fig2.update_layout(
            title=f"IV Rank ({tenor}-day, 100-day lookback)",
            xaxis_title="Date", yaxis_title="Rank (0-100)",
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    available_dates = daily.index.tolist()
    selected_date = st.select_slider(
        "Select Date",
        options=available_dates,
        value=available_dates[-1],
        format_func=lambda d: str(d.date()),
    )
    ts = pd.Timestamp(selected_date)
    S = universe.spot(ts)
    tenors = st.multiselect("Tenors (days)", [5, 9, 14, 21, 30], default=[5, 14, 30])
    if tenors:
        fig = iv_smile(surface, ts, universe, tenors_days=tenors)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("SPY Spot", f"${universe.spot(ts):.2f}")
    col2.metric("VIX (30d ATM IV)", f"{universe.iv_atm_30d(ts):.1%}")
    col3.metric("CBOE SKEW", f"{universe.skew_index(ts):.1f}")
