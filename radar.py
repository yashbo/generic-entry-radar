"""Shared data assembly for the app and batch scripts: the scored entry table and per-drug facts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import attractiveness
import backtest
import entry
import orange_book
import partd
import piv

PIV_COLS = ["piv_first_date", "piv_max_andas", "piv_status_180", "piv_ftf_approval", "piv_ftf_marketing",
            "piv_last_qualifying_patent"]
MODEL_NOTE = ("Statistical estimate that a generic is approved within 2 years after listed protection ends, "
              "trained on 2010-2022 Orange Book outcomes. Driven mostly by dosage form, route, "
              "patent count and Medicare Part D sales.")


def build_table(data_dir: Path, today: pd.Timestamp, with_model: bool = True):
    """Orange Book + entry estimates + model likelihood + Part D spend + Paragraph IV columns."""
    ob = orange_book.load(Path(data_dir))
    table = entry.build_entry_table(ob, today=today)
    bundle = attractiveness.load_model() if with_model and attractiveness.MODEL_PATH.exists() else None
    # Score as of the data's own date, not the wall clock: otherwise the likelihood drifts a point or two
    # every few weeks with no new data, and stored AI answers that quote it fail the display-time check.
    edition = orange_book.edition_date(ob)
    model_as_of = min(today, edition) if edition is not None else today
    if bundle:
        # Model estimate (0-100) that a generic arrives within 2 yrs of protection ending.
        # Features such as active patent counts also depend on the date, so score a table built as of then.
        as_of_table = table if model_as_of == today else entry.build_entry_table(ob, today=model_as_of)
        p = attractiveness.score(bundle, ob, as_of_table, model_as_of) * 100
        in_cohort = table["status"].isin(backtest.COHORT)  # only drugs without a generic today are scored
        table["p_generic"] = table["appl_no"].map(p).round(0).where(in_cohort)
    else:
        table["p_generic"] = float("nan")
    long = partd.load_long()
    sp = partd.spend_features(table, long, today)
    table["partd_spend_m"] = ((10 ** sp["log_brand_spend"] - 1) / 1e6).round(1)
    pv = piv.load()
    if pv is not None:
        table = table.merge(pv, on="appl_no", how="left")
    else:
        for c in PIV_COLS:
            table[c] = None
    table.attrs["spend_year"] = partd.spend_year(long, today) if long is not None else None
    # Stored as ISO strings: Streamlit serialises DataFrame attrs as JSON.
    piv_date = piv.list_date()
    table.attrs["edition_date"] = edition.strftime("%Y-%m-%d") if edition is not None else None
    table.attrs["model_as_of"] = model_as_of.strftime("%Y-%m-%d")
    table.attrs["piv_date"] = piv_date.strftime("%Y-%m-%d") if piv_date is not None else None
    return ob, table


def _d(v):
    return v.strftime("%Y-%m-%d") if pd.notna(v) else None


def full_facts(ob, table: pd.DataFrame, appl_no: str) -> dict[str, Any]:
    """Everything the explainer is given for one drug."""
    facts = entry.drug_facts(ob, table, appl_no)
    row = table.loc[table["appl_no"] == appl_no].iloc[0]
    if pd.notna(row["p_generic"]):
        facts["model_generic_likelihood_pct"] = int(row["p_generic"])
        facts["model_note"] = MODEL_NOTE
    if pd.notna(row["piv_first_date"]):
        facts["paragraph_iv_challenge"] = {
            "first_filed": _d(row["piv_first_date"]),
            "number_of_challenging_generics": None if pd.isna(row["piv_max_andas"]) else int(row["piv_max_andas"]),
            "180_day_exclusivity_status": row["piv_status_180"] or None,
            "first_challenger_approved": _d(row["piv_ftf_approval"]),
            "first_challenger_launched": _d(row["piv_ftf_marketing"]),
            "last_qualifying_patent_expires": _d(row["piv_last_qualifying_patent"]),
        }
    spend_m = row["partd_spend_m"]
    if pd.notna(spend_m) and spend_m > 0:  # zero means no Part D sales: leave it out
        facts[f"medicare_part_d_brand_spend_{table.attrs.get('spend_year')}_usd_millions"] = float(spend_m)
    return facts
