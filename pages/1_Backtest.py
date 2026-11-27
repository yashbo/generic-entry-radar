"""Backtest page: how well did past Orange Book editions predict generic entry?"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backtest  # noqa: E402

st.set_page_config(page_title="Backtest · Generic Entry Radar", page_icon="📡", layout="wide")
st.title("Backtest")
st.caption(
    "Each archived Orange Book edition predicts entry for drugs with no generic at that time. "
    "Truth is the first generic (ANDA) approval afterwards, from the current Orange Book. "
    f"Timing is scored only once a prediction is {backtest.MATURITY_YEARS}+ years old, "
    "so early, late and 'never' outcomes are all visible."
)

hist = backtest.HIST_DIR
saved_rows = backtest.OUT_DIR / "backtest_rows.csv"
DATES = ["as_of", "observed_until", "pred_all_listed", "pred_carveout", "pred_compound", "actual_first_generic"]


@st.cache_data(show_spinner="Running backtest over archived editions (about 30 s) ...")
def run(_mtime: float):
    return backtest.run()


@st.cache_data
def load_saved(_mtime: float):
    # Deployed copies ship the precomputed results instead of 38 MB of archived editions.
    rows = pd.read_csv(saved_rows, parse_dates=DATES, dtype={"appl_no": str})
    return rows, backtest.summarize(rows)


if hist.exists() and any(p.is_dir() for p in hist.iterdir()):
    rows, summary = run(max(p.stat().st_mtime for p in hist.iterdir()))
elif saved_rows.exists():
    rows, summary = load_saved(saved_rows.stat().st_mtime)
else:
    st.warning(
        "No archived editions found. Run:\n\n"
        "```\npython fetch_history.py\n```\n"
        "then reload this page."
    )
    st.stop()

NAMES = {
    "all_listed": "All listed protection",
    "carveout": "With carve-outs",
    "compound": "Compound patent + exclusivity only",
}
predictor = st.radio("Predictor", list(NAMES), format_func=NAMES.get, horizontal=True)

rows = rows.assign(outcome=backtest.classify(rows, predictor))
scored = rows[~rows["prior_anda"]]
matured = scored[scored["outcome"] != "Not matured"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predictions scored", f"{len(matured):,}", help="Matured predictions across all editions")
share = matured["outcome"].value_counts(normalize=True).mul(100)
c2.metric("Within ±1 year", f"{share.get('Within ±1y', 0):.0f}%")
c3.metric("Generic > 1 yr early", f"{share.get('Generic >1y early', 0):.0f}%")
c4.metric("No generic ever came", f"{share.get('No generic yet', 0):.0f}%")

# ------------------------------------------------------------ outcome mix
st.subheader("What happened after the predicted date")
mix = (
    matured.groupby([matured["as_of"].dt.year.astype(str), "outcome"]).size().rename("n").reset_index()
    .rename(columns={"as_of": "edition"})
)
mix["pct"] = mix["n"] / mix.groupby("edition")["n"].transform("sum") * 100
fig = px.bar(
    mix, x="edition", y="pct", color="outcome", category_orders={"outcome": backtest.OUTCOMES[:4]},
    color_discrete_map={
        "Generic >1y early": "#d62728", "Within ±1y": "#2ca02c", "Generic >1y late": "#ff7f0e", "No generic yet": "#b0b0b0",
    },
    labels={"edition": "Edition", "pct": "% of matured predictions", "outcome": ""},
    hover_data={"n": True, "pct": ":.1f"},
)
fig.update_xaxes(type="category")
fig.update_layout(height=360, margin=dict(t=10, b=10))
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------ error distribution
st.subheader("Timing error when a generic did arrive")
err = matured.dropna(subset=[f"err_{predictor}"])
fig = px.histogram(
    err, x=f"err_{predictor}", nbins=60,
    labels={f"err_{predictor}": "Actual first generic minus predicted (years; negative = early)"},
)
fig.add_vline(x=0, line_dash="dash", line_color="gray")
fig.update_layout(height=300, margin=dict(t=10, b=10), yaxis_title="Drugs")
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------ predictor comparison
st.subheader("Predictor comparison by edition")
st.caption(
    f"precision/recall_{backtest.HORIZON_YEARS}y: will a generic be approved within {backtest.HORIZON_YEARS} years of the edition? "
    "Only shown where that window is fully observed."
)
show = summary.copy()
show["predictor"] = show["predictor"].map(NAMES)
st.dataframe(show, hide_index=True, width="stretch")

# ------------------------------------------------------------ drill-down
st.subheader("Drug-level results")
f1, f2 = st.columns(2)
ed = f1.selectbox("Edition", sorted(rows["as_of"].dt.date.unique(), reverse=True))
oc = f2.multiselect("Outcome", backtest.OUTCOMES, default=backtest.OUTCOMES[:3])
d = rows[(rows["as_of"].dt.date == ed) & rows["outcome"].isin(oc)]
cols = ["trade_name", "ingredient", "df_route", "main_barrier", f"pred_{predictor}", "actual_first_generic", f"err_{predictor}", "outcome"]
st.dataframe(
    d[cols].sort_values(f"err_{predictor}"),
    hide_index=True, width="stretch", height=min(420, 38 + 35 * max(len(d), 1)),
    column_config={
        f"pred_{predictor}": st.column_config.DateColumn("Predicted", format="YYYY-MM-DD"),
        "actual_first_generic": st.column_config.DateColumn("First generic", format="YYYY-MM-DD"),
        f"err_{predictor}": st.column_config.NumberColumn("Error (yrs)", format="%.1f"),
    },
)
st.download_button("Download drug-level CSV", rows.to_csv(index=False), "backtest_rows.csv")
