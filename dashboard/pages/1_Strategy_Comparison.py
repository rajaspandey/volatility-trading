"""Strategy Comparison page — compare equity curves side-by-side."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.set_page_config(page_title="Strategy Comparison", page_icon="⚖️", layout="wide")
st.title("Strategy Comparison")

from src.reporting.metrics import compute_metrics
from src.reporting.plots import equity_curves, drawdown_chart, monthly_returns_heatmap

STRATEGY_LABELS = {
    "straddle":      "Short Straddle",
    "strangle_ss0":  "Strangle (ATM)",
    "strangle_ss5":  "Strangle (ss=5%)",
    "strangle_ss10": "Strangle (ss=10%)",
    "skew_arb":      "Skew Arb",
}

# Load saved equity curves
eq_dir = Path("data/processed")
available = {}
for k, label in STRATEGY_LABELS.items():
    fp = eq_dir / f"equity_{k}.parquet"
    if fp.exists():
        available[k] = pd.read_parquet(fp)

if not available:
    st.warning("No backtest results found. Run `python scripts/run_all.py` first.")
    st.stop()

# Strategy selector
selected = st.multiselect(
    "Select strategies to compare",
    list(available.keys()),
    default=list(available.keys()),
    format_func=lambda k: STRATEGY_LABELS[k],
)
if not selected:
    st.info("Select at least one strategy.")
    st.stop()

subset = {k: available[k] for k in selected}

# KPI row
cap = 1_000_000.0
cols = st.columns(len(subset))
for col, (k, eq) in zip(cols, subset.items()):
    m = compute_metrics(eq, cap)
    col.metric(
        STRATEGY_LABELS[k],
        f"{m['ann_return']:+.2%}",
        f"Sharpe {m['sharpe']:.2f}",
    )

st.plotly_chart(
    equity_curves({STRATEGY_LABELS[k]: eq for k, eq in subset.items()},
                  "Equity Curves Comparison"),
    use_container_width=True,
)

# Full metrics table
st.subheader("Performance Metrics")
rows = []
for k, eq in subset.items():
    m = compute_metrics(eq, cap)
    rows.append({
        "Strategy": STRATEGY_LABELS[k],
        "Ann. Return": f"{m['ann_return']:+.2%}",
        "Sharpe": f"{m['sharpe']:.2f}",
        "Max DD": f"{m['max_drawdown']:.2%}",
        "Ann. Vol": f"{m['ann_vol']:.2%}",
        "Win Rate": f"{m['win_rate']:.2%}",
        "Trading Days": m["n_trading_days"],
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Monthly heatmaps
st.subheader("Monthly Return Heatmaps")
for k, eq in subset.items():
    if eq is None or eq.empty or "equity" not in eq.columns:
        continue
    fig = monthly_returns_heatmap(eq["equity"], title=STRATEGY_LABELS[k])
    st.plotly_chart(fig, use_container_width=True)
