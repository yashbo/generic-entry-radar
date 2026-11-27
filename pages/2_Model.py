"""Model page: how the generic-attractiveness model was evaluated."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import attractiveness  # noqa: E402

st.set_page_config(page_title="Model · Generic Entry Radar", page_icon="📡", layout="wide")
st.title("Generic attractiveness model")
st.caption(
    f"Predicts whether a generic is approved within {attractiveness.GENERIC_WINDOW_YEARS} years after a drug's listed "
    "protection ends. Trained on matured backtest cases from the 2010-2022 Orange Book editions, using only "
    "features known at each edition's date."
)

if not attractiveness.METRICS_PATH.exists():
    st.warning("No model yet. Run `python attractiveness.py` (about 1 minute), then reload.")
    st.stop()

m = json.loads(attractiveness.METRICS_PATH.read_text())
final = attractiveness.FINAL_MODEL
LABELS = {
    "logistic_regression": "Logistic reg. (final)",
    "gradient_boosting": "Gradient boosting",
    "baseline_form_and_route": "Form + route only",
    "baseline_spend_only": "Part D spend only",
    "baseline_generic_elsewhere": "Generic elsewhere",
    "base_rate": "Base rate",
}

t = m["temporal"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Training cases", f"{m['n_rows']:,}", help=f"{m['n_drugs']:,} distinct drugs")
c2.metric("Generic arrived", f"{m['positive_rate']:.0%}")
c3.metric("AUC, unseen later drugs", f"{t[final]['roc_auc']:.2f}", help="Trained on editions before 2016, tested on 2016+ drugs never seen in training")
c4.metric("Brier score, unseen later drugs", f"{t[final]['brier']:.3f}", help="Lower is better; base rate scores about 0.24")

st.subheader("Evaluation")
left, right = st.columns(2)


def table(d: dict) -> pd.DataFrame:
    rows = {LABELS.get(k, k): v for k, v in d.items() if isinstance(v, dict)}
    df = pd.DataFrame(rows).T.rename(columns={"roc_auc": "AUC", "avg_precision": "Avg prec.", "brier": "Brier"})
    return df.style.format("{:.3f}")


left.markdown("**Grouped 5-fold CV** (a drug never appears in both train and test)")
left.table(table(m["cv"]))
right.markdown(f"**Temporal holdout**: train < 2016 ({t['n_train']:,} cases), test 2016+ unseen drugs ({t['n_test']:,} cases)")
right.table(table(t))
st.info(
    "Dosage form and route carry the most signal on their own. Patent counts, Medicare Part D sales and Paragraph IV "
    "challenges add to it. Spend alone is weak (clinic-given injectables barely appear in Part D) but helps in combination."
)
versions = [
    ("1. Orange Book features", attractiveness.MODEL_DIR / "metrics_no_spend.json"),
    ("2. Add Medicare Part D sales", attractiveness.MODEL_DIR / "metrics_no_piv.json"),
    ("3. Add Paragraph IV challenges (current)", attractiveness.METRICS_PATH),
]
hist = []
for label, path in versions:
    if path.exists():
        mm = json.loads(path.read_text())
        hist.append({"Version": label, "CV AUC": mm["cv"][final]["roc_auc"], "CV Brier": mm["cv"][final]["brier"],
                     "Temporal AUC": mm["temporal"][final]["roc_auc"], "Temporal Brier": mm["temporal"][final]["brier"]})
if len(hist) > 1:
    st.markdown("**Model versions** (final model; lower Brier is better)")
    st.table(pd.DataFrame(hist).set_index("Version").style.format("{:.3f}"))
    st.caption(
        "Why the Paragraph IV step barely moves the temporal numbers: that test uses only drugs never seen in training, "
        "i.e. newly protected drugs, and none of them had a patent challenge on file yet (challenges usually come years "
        "after approval). So the temporal test cannot measure that feature; grouped CV, which does include challenged "
        "drugs, shows the gain. In the training data, 81% of drugs already under challenge got a generic, versus 24% without."
    )

left, right = st.columns(2)
with left:
    st.markdown("**What drives predictions** (drop in AUC when a feature is shuffled, temporal test)")
    imp = pd.DataFrame(m["importance"]).head(10)
    fig = px.bar(imp[::-1], x="auc_drop", y="feature", orientation="h", error_x="std",
                 labels={"auc_drop": "AUC drop", "feature": ""})
    fig.update_layout(height=360, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
with right:
    st.markdown("**Calibration** (out-of-fold, deciles): points on the diagonal mean the % is trustworthy")
    cal = pd.DataFrame(m["calibration"])
    fig = go.Figure([
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect"),
        go.Scatter(x=cal["predicted"], y=cal["actual"], mode="lines+markers", name="Model",
                   text=cal["n"], hovertemplate="predicted %{x:.2f}<br>actual %{y:.2f}<br>n=%{text}"),
    ])
    fig.update_layout(height=360, margin=dict(t=10, b=10), xaxis_title="Predicted", yaxis_title="Actual",
                      xaxis_range=[0, 1], yaxis_range=[0, 1])
    st.plotly_chart(fig, width="stretch")

st.subheader("Generic rate by dosage form in the training data")
data_path = attractiveness.MODEL_DIR / "training_data.csv"
if data_path.exists():
    d = pd.read_csv(data_path)
    g = d.groupby("form_group")["y"].agg(["mean", "size"]).reset_index().sort_values("mean")
    fig = px.bar(g, x="mean", y="form_group", orientation="h", text="size",
                 labels={"mean": "Share that got a generic", "form_group": "", "size": "cases"})
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig.update_layout(height=340, margin=dict(t=10, b=10), xaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

st.caption(
    "Limits: sales are Medicare Part D only (no commercial, Medicaid or Part B), spend for years before 2011 is "
    "unavailable (the 2010 edition has none), ANDA approval is not launch, and drugs are matched by name."
)
