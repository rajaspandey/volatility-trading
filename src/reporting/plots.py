"""ResultPlotter: reusable Plotly figures for the dashboard and notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def equity_curves(
    results: dict[str, pd.DataFrame],
    title: str = "Portfolio Equity Curves",
    initial_capital: float = 1_000_000.0,
) -> go.Figure:
    """Overlay equity curves from multiple strategy runs."""
    fig = go.Figure()
    for name, eq in results.items():
        if eq is None or eq.empty:
            continue
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq["equity"],
            mode="lines", name=name, line=dict(width=2),
        ))
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                  annotation_text="Initial capital")
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Equity ($)",
        hovermode="x unified", height=450,
    )
    return fig


def drawdown_chart(equity: pd.Series, title: str = "Drawdown") -> go.Figure:
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy", mode="lines",
        line=dict(color="#EF553B", width=1),
        name="Drawdown %",
    ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Drawdown (%)",
        height=280,
    )
    return fig


def iv_rv_spread(
    daily: pd.DataFrame,
    tenor: str = "30",
    title: str | None = None,
) -> go.Figure:
    """IV vs RV with spread, for the given tenor ('5' or '30')."""
    iv_col  = f"iv_{tenor}"
    rv_col  = f"rv_{tenor}"
    diff_col = f"iv_rv_diff_{tenor}"
    label = f"{tenor}-day"
    t = title or f"IV vs Realized Vol ({label})"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=[f"IV and RV ({label})", "IV − RV Spread"],
    )
    if iv_col in daily.columns:
        fig.add_trace(go.Scatter(
            x=daily.index, y=daily[iv_col] * 100,
            mode="lines", name=f"VIX ({label} ATM IV)",
            line=dict(color="#636EFA"),
        ), row=1, col=1)
    if rv_col in daily.columns:
        fig.add_trace(go.Scatter(
            x=daily.index, y=daily[rv_col] * 100,
            mode="lines", name=f"RV {label}",
            line=dict(color="#EF553B"),
        ), row=1, col=1)
    if diff_col in daily.columns:
        spread = daily[diff_col] * 100
        fig.add_trace(go.Bar(
            x=daily.index, y=spread,
            name="IV − RV",
            marker_color=np.where(spread >= 0, "#00CC96", "#EF553B"),
        ), row=2, col=1)
    fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray")
    fig.update_layout(title=t, height=480, hovermode="x unified")
    return fig


def iv_smile(
    surface,
    ts: pd.Timestamp,
    universe,
    tenors_days: list[int] | None = None,
) -> go.Figure:
    """Plot the IV smile for multiple tenors on a given date."""
    if tenors_days is None:
        tenors_days = [5, 9, 14, 21, 30]
    S = universe.spot(ts)
    r = universe.rate(ts)
    strike_pcts = np.arange(0.85, 1.16, 0.01)
    strikes = strike_pcts * S

    fig = go.Figure()
    for T_days in tenors_days:
        T = T_days / 252.0
        ivs = []
        for K in strikes:
            flag = "p" if K <= S else "c"
            ivs.append(surface.iv_for_strike(ts, K, T, flag) * 100)
        fig.add_trace(go.Scatter(
            x=strike_pcts * 100, y=ivs,
            mode="lines", name=f"{T_days}d",
        ))
    fig.add_vline(x=100.0, line_dash="dash", line_color="gray",
                  annotation_text="ATM")
    fig.update_layout(
        title=f"IV Smile — {ts.date()}",
        xaxis_title="Strike (% of Spot)",
        yaxis_title="Implied Vol (%)",
        hovermode="x unified",
        height=420,
    )
    return fig


def monthly_returns_heatmap(equity: pd.Series, title: str = "Monthly Returns") -> go.Figure:
    """Heatmap of month × year returns."""
    returns = equity.pct_change().dropna()
    monthly = returns.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    df = monthly.to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot(index="year", columns="month", values="ret") * 100
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[month_names[m - 1] for m in pivot.columns],
        y=pivot.index.astype(str),
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(pivot.values, 2),
        texttemplate="%{text:.1f}%",
        showscale=True,
    ))
    fig.update_layout(title=title, height=max(200, len(pivot) * 60 + 80))
    return fig
