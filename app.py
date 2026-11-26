"""Generic Entry Radar: Streamlit dashboard.

Run from the project folder:
    python -m streamlit run generic_entry_radar/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import attractiveness  # noqa: E402
import entry  # noqa: E402
import explain  # noqa: E402
import orange_book  # noqa: E402
import partd  # noqa: E402
import piv  # noqa: E402

REAL_DIR = Path(os.environ.get("RADAR_DATA_DIR", orange_book.DEFAULT_DIR))
DEMO_DIR = ROOT / "tests" / "fixtures" / "orange_book"
TODAY = pd.Timestamp.today().normalize()

STATUS_ORDER = [
    "Protected", "Carve-out possible now", "Open: no listed barrier, no generic",
    "Generic approved, patents still listed", "Generic approved",
]

st.set_page_config(page_title="Generic Entry Radar", page_icon="📡", layout="wide")


@st.cache_data(show_spinner="Loading Orange Book ...")
def load(data_dir: str, model_mtime: float):
    ob = orange_book.load(Path(data_dir))
    table = entry.build_entry_table(ob, today=TODAY)
    bundle = attractiveness.load_model() if model_mtime else None
    if bundle:
        # Model estimate (0-100) that a generic arrives within 2 yrs of protection ending.
        table["p_generic"] = table["appl_no"].map(attractiveness.score(bundle, ob, table, TODAY) * 100).round(0)
    else:
        table["p_generic"] = float("nan")
    long = partd.load_long()
    sp = partd.spend_features(table, long, TODAY)
    table["partd_spend_m"] = ((10 ** sp["log_brand_spend"] - 1) / 1e6).round(1)
    spend_year = partd.spend_year(long, TODAY) if long is not None else None
    pv = piv.load()
    if pv is not None:
        table = table.merge(pv, on="appl_no", how="left")
    else:
        for c in ["piv_first_date", "piv_max_andas", "piv_status_180", "piv_ftf_approval", "piv_ftf_marketing", "piv_last_qualifying_patent"]:
            table[c] = None
    table.attrs["spend_year"] = spend_year
    return ob, table


# ---------------------------------------------------------------- data source
if (REAL_DIR / "products.txt").exists():
    data_dir, demo = REAL_DIR, False
else:
    data_dir, demo = DEMO_DIR, True

model_path = attractiveness.MODEL_PATH
ob, table = load(str(data_dir), model_path.stat().st_mtime if model_path.exists() else 0.0)
has_model = table["p_generic"].notna().any()

st.title("📡 Generic Entry Radar")
st.caption(
    "Earliest generic entry estimated from FDA Orange Book patents and exclusivities. "
    "Listed protection only; litigation, settlements and unlisted patents can move real dates. Not legal advice."
)
if demo:
    st.warning(
        "Showing **synthetic demo data**. Download the Orange Book zip from "
        "[FDA](https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files), "
        "save it in `generic_entry_radar/data/`, run `python generic_entry_radar/fetch_data.py`, then rerun the app."
    )

# ---------------------------------------------------------------- filters
with st.sidebar:
    st.header("Filters")
    q = st.text_input("Search trade name, ingredient or applicant")
    statuses = st.multiselect("Status", STATUS_ORDER, default=STATUS_ORDER[:3])
    routes = sorted({r.split(";")[-1].strip() for v in table["df_route"].dropna() for r in v.split("; ")})
    route_sel = st.multiselect("Route", routes)
    horizon = st.slider("Entry within (years)", 1, 20, 10)
    only_piv = st.checkbox("Only drugs facing a Paragraph IV challenge",
                           help="A generic maker has certified that a listed patent is invalid or not infringed.")
    min_p = st.slider("Min. generic likelihood (%)", 0, 100, 0, disabled=not has_model,
                      help="Model estimate that a generic is approved within 2 years of protection ending. "
                           "Only drugs without a generic are scored.")
    st.divider()
    ok, msg = explain.available()
    st.caption(("🟢 LLM explainer: " + msg) if ok else
               "⚪ AI explanations run locally with Ollama; this copy shows verified briefs instead.")

view = table[table["status"].isin(statuses)]
if q:
    ql = q.lower()
    view = view[
        view[["trade_name", "ingredient", "applicant"]].fillna("").apply(lambda c: c.str.lower().str.contains(ql, regex=False)).any(axis=1)
    ]
if route_sel:
    view = view[view["df_route"].fillna("").apply(lambda v: any(r in v for r in route_sel))]
if has_model and min_p > 0:
    view = view[view["p_generic"] >= min_p]
if only_piv:
    view = view[view["piv_first_date"].notna()]

# ---------------------------------------------------------------- KPIs
protected = table[table["status"] == "Protected"]
soon = protected[protected["entry_all_listed"] <= TODAY + pd.DateOffset(years=3)]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Active NDAs", f"{len(table):,}")
c2.metric("Protected", f"{len(protected):,}")
c3.metric("Ends ≤ 3 yrs", f"{len(soon):,}")
c4.metric("…generic likely", f"{(soon['p_generic'] >= 50).sum():,}" if has_model else "n/a",
          help="Of those, how many the model gives ≥ 50% chance of a generic within 2 years. "
               "Train it with: python generic_entry_radar/attractiveness.py")
c5.metric("Open, no generic", f"{(table['status'] == 'Open: no listed barrier, no generic').sum():,}")
n_challenged = protected["piv_first_date"].notna().sum()
if n_challenged:
    st.caption(f"⚔️ {n_challenged:,} of {len(protected):,} protected drugs already face a Paragraph IV patent challenge, "
               "so generics may arrive before the listed dates.")

# ---------------------------------------------------------------- cliff chart
st.subheader("Protection cliff")
cliff = view[(view["status"] == "Protected") & (view["entry_all_listed"] <= TODAY + pd.DateOffset(years=horizon))].copy()
if cliff.empty:
    st.info("No protected drugs match the filters in this window.")
else:
    cliff["year"] = cliff["entry_all_listed"].dt.year.astype(str)
    fig = px.histogram(
        cliff, x="year", color="main_barrier", barmode="stack",
        category_orders={"year": sorted(cliff["year"].unique())},
        labels={"year": "Year all listed protection ends", "main_barrier": "Last barrier"},
    )
    fig.update_xaxes(type="category")
    fig.update_layout(yaxis_title="NDAs", height=340, margin=dict(t=10, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- table
st.subheader(f"Drugs ({len(view):,})")
cols = {
    "trade_name": "Trade name", "ingredient": "Ingredient", "df_route": "Form; route", "applicant": "Applicant",
    "status": "Status", "main_barrier": "Last barrier", "entry_all_listed": "Entry (all listed)",
    "entry_with_carveout": "Entry (carve-out)", "years_to_entry": "Years", "p_generic": "Generic likelihood",
    "partd_spend_m": "Part D spend ($M)", "piv_first_date": "PIV filed", "piv_max_andas": "PIV challengers",
    "n_active_patents": "Active patents",
    "active_exclusivities": "Active exclusivities", "n_generic_makers": "Generic makers", "appl_no": "NDA",
}
st.dataframe(
    view[list(cols)].rename(columns=cols),
    hide_index=True, width="stretch", height=min(400, 38 + 35 * max(len(view), 1)),
    column_config={
        "Entry (all listed)": st.column_config.DateColumn(format="YYYY-MM-DD"),
        "Entry (carve-out)": st.column_config.DateColumn(format="YYYY-MM-DD"),
        "PIV filed": st.column_config.DateColumn(format="YYYY-MM-DD", help="First Paragraph IV patent challenge"),
        "Generic likelihood": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "Part D spend ($M)": st.column_config.NumberColumn(format="%.1f",
            help=f"Medicare Part D gross spend on this brand in {table.attrs.get('spend_year')}"),
    },
)
st.download_button("Download CSV", view[list(cols)].rename(columns=cols).to_csv(index=False), "generic_entry_radar.csv")

# ---------------------------------------------------------------- drug detail
st.subheader("Drug deep dive")
if view.empty:
    st.stop()
labels = (view["trade_name"] + " — " + view["ingredient"] + " (NDA " + view["appl_no"] + ")").tolist()
pick = st.selectbox("Choose a drug", labels)
appl_no = view["appl_no"].iloc[labels.index(pick)]
facts = entry.drug_facts(ob, table, appl_no)
p_gen = table.loc[table["appl_no"] == appl_no, "p_generic"].iloc[0]
if pd.notna(p_gen):
    facts["model_generic_likelihood_pct"] = int(p_gen)
    facts["model_note"] = ("Statistical estimate that a generic is approved within 2 years after listed protection ends, "
                           "trained on 2010-2022 Orange Book outcomes. Driven mostly by dosage form, route, "
                           "patent count and Medicare Part D sales.")
spend_m = table.loc[table["appl_no"] == appl_no, "partd_spend_m"].iloc[0]
prow = table.loc[table["appl_no"] == appl_no].iloc[0]


def _d(v):
    return v.strftime("%Y-%m-%d") if pd.notna(v) else None


has_piv = pd.notna(prow["piv_first_date"])
if has_piv:
    facts["paragraph_iv_challenge"] = {
        "first_filed": _d(prow["piv_first_date"]),
        "number_of_challenging_generics": None if pd.isna(prow["piv_max_andas"]) else int(prow["piv_max_andas"]),
        "180_day_exclusivity_status": prow["piv_status_180"] or None,
        "first_challenger_approved": _d(prow["piv_ftf_approval"]),
        "first_challenger_launched": _d(prow["piv_ftf_marketing"]),
        "last_qualifying_patent_expires": _d(prow["piv_last_qualifying_patent"]),
    }
if pd.notna(spend_m) and spend_m > 0:  # zero means no Part D sales: leave it out
    facts[f"medicare_part_d_brand_spend_{table.attrs.get('spend_year')}_usd_millions"] = float(spend_m)

st.markdown(f"**{facts['trade_name']}** · {facts['applicant']} · status: **{facts['status']}** · last barrier: {facts['main_barrier']}")
a, b, c = st.columns(3)
b.metric("Entry (all listed)", facts["estimated_entry_all_listed"] or "n/a")
c.metric("Entry (carve-out)", facts["estimated_entry_with_carveout"] or "n/a")
a.metric("Generic likelihood", f"{int(p_gen)}%" if pd.notna(p_gen) else "n/a",
         help="Only scored for drugs that have no generic yet.")
d, e, _ = st.columns(3)
d.metric("Generic makers", facts["n_generic_makers"])
e.metric(f"Part D spend {table.attrs.get('spend_year') or ''}", f"${spend_m:,.0f}M" if pd.notna(spend_m) else "n/a",
         help="Medicare Part D gross spend on this brand (retail Medicare only)")
if has_piv:
    pi = facts["paragraph_iv_challenge"]
    st.markdown("**⚔️ Paragraph IV patent challenge**")
    w, x, y, z = st.columns([1.5, 1, 1.2, 1.5])
    w.metric("First filed", pi["first_filed"])
    x.metric("Challengers", pi["number_of_challenging_generics"] or "n/a", help="Substantially complete ANDAs with a PIV certification")
    y.metric("180-day status", pi["180_day_exclusivity_status"] or "Pending",
             help="Eligible / Extinguished / Deferred etc. Blank on FDA's list usually means no decision yet.")
    z.metric("Generic launched", pi["first_challenger_launched"] or "Not yet",
             help="First commercial marketing by the first applicant to file a challenge")
else:
    st.caption("No Paragraph IV challenge on FDA's list for this NDA.")

rows = []
for p in facts["patents"]:
    kind = "Compound" if p["substance"] else "Formulation" if p["product"] else "Method of use" if p["use_codes"] else "Other"
    if p["delisted"]:
        kind += " (delisting requested)"
    rows.append({"item": f"Patent {p['patent_no']}", "type": kind, "expires": p["expires"], "detail": p["use_codes"]})
for ex in facts["exclusivities"]:
    rows.append({"item": f"Exclusivity {ex['exclusivity_code']}", "type": "Exclusivity", "expires": ex["expires"], "detail": ex["meaning"]})
if has_piv:
    rows.append({"item": "Paragraph IV filed", "type": "Patent challenge", "expires": facts["paragraph_iv_challenge"]["first_filed"],
                 "detail": f"{facts['paragraph_iv_challenge']['number_of_challenging_generics'] or '?'} challenger(s)"})

if rows:
    tl = pd.DataFrame(rows).dropna(subset=["expires"])
    tl["expires"] = pd.to_datetime(tl["expires"])
    fig = px.scatter(tl, x="expires", y="item", color="type", hover_data=["detail"], height=120 + 28 * len(tl))
    fig.update_traces(marker=dict(size=12, symbol="diamond"))
    fig.add_vline(x=TODAY.to_pydatetime(), line_dash="dash", line_color="gray")
    fig.update_layout(yaxis_title="", xaxis_title="Date (dashed line = today)", margin=dict(t=10, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
else:
    st.info("No patents or exclusivities listed for this NDA.")

with st.expander("Products and raw facts sent to the LLM"):
    st.json(facts)

st.markdown("#### AI explanation")
if not ok:
    # No local LLM (e.g. the hosted demo): show the deterministic, verified brief instead.
    st.caption("AI-written explanations run locally with Ollama. Without it, here is the verified brief the model "
               "would be given, computed directly from the data.")
    st.markdown(explain.brief_as_markdown(facts))
elif st.button("Explain this drug's generic-entry outlook", type="primary"):
    try:
        with st.spinner(f"Writing and fact-checking with {explain.MODEL} ..."):
            text, report = explain.explain_checked(facts)
        if report["fallback"]:
            st.warning("The model's explanation failed the fact check twice, so here is the verified brief instead: "
                       + "; ".join(report["issues"][:3]))
        else:
            st.caption(f"✅ Fact-checked: every patent, date and code is in the source data, the bottom-line date is "
                       f"correct and patent types match (attempt {report['attempts']}).")
        st.markdown(text)
    except Exception as exc:  # noqa: BLE001
        st.error(f"LLM call failed: {exc}")
    with st.expander("Brief given to the model"):
        st.text(explain.build_brief(facts))
