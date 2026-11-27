"""Backtest generic-entry predictions against what actually happened.

For each archived Orange Book edition (as-of date E):
  * cohort  = NDAs with no approved generic at E (status Protected / Carve-out possible now)
  * predict = entry dates computed from that edition alone:
      - all_listed : every listed patent and exclusivity
      - carveout   : ignores method-of-use patents and label-specific exclusivities
      - compound   : only drug-substance patents and non-label-specific exclusivities
  * actual  = first ANDA approval on/after E for the same ingredient + dosage form/route,
              taken from the current Orange Book (which keeps discontinued ANDAs).

Caveats: ANDA approval is not commercial launch (settlements can delay launch), and
ingredient names occasionally change between editions, so a few matches are missed.

Run: python backtest.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import entry
import orange_book

ROOT = Path(__file__).resolve().parent
HIST_DIR = ROOT / "data" / "history"
OUT_DIR = ROOT / "data" / "backtest"
PREDICTORS = ("all_listed", "carveout", "compound")
COHORT = {"Protected", "Carve-out possible now"}
HORIZON_YEARS = 3


def _nda_keys(ob: orange_book.OrangeBook) -> pd.DataFrame:
    p = ob.products
    p = p[(p["appl_type"] == "N") & p["type"].isin(entry.ACTIVE_TYPES)]
    return pd.DataFrame({"appl_no": p["appl_no"], "key": entry._key(p)}).drop_duplicates()


def _anda_approvals(current: orange_book.OrangeBook) -> pd.DataFrame:
    p = current.products
    p = p[(p["appl_type"] == "A") & p["approval_date"].notna()]
    return pd.DataFrame({"key": entry._key(p), "approval_date": p["approval_date"]})


def backtest_edition(
    edition: orange_book.OrangeBook,
    as_of: pd.Timestamp,
    current: orange_book.OrangeBook,
    observed_until: pd.Timestamp,
) -> pd.DataFrame:
    """One row per cohort NDA with predictions, actual outcome and errors (in years)."""
    t = entry.build_entry_table(edition, today=as_of)
    t = t[t["status"].isin(COHORT)].copy()
    t["pred_all_listed"] = t["entry_all_listed"]
    t["pred_carveout"] = t["entry_with_carveout"]
    t["pred_compound"] = t[["substance_patent_expiry", "hard_exclusivity_expiry"]].max(axis=1)

    hits = _nda_keys(edition).merge(_anda_approvals(current), on="key")
    hits = hits[hits["appl_no"].isin(t["appl_no"])]
    after = hits[hits["approval_date"] >= as_of].groupby("appl_no")["approval_date"].min()
    before = hits[hits["approval_date"] < as_of].groupby("appl_no").size()

    t["actual_first_generic"] = t["appl_no"].map(after)
    # A generic approved before E that the edition did not list as active usually means
    # it was discontinued at E or the names did not line up; keep it out of error stats.
    t["prior_anda"] = t["appl_no"].isin(before.index)
    t["generic_observed"] = t["actual_first_generic"].notna()
    t["as_of"] = as_of
    t["observed_until"] = observed_until

    for name in PREDICTORS:
        t[f"err_{name}"] = (t["actual_first_generic"] - t[f"pred_{name}"]).dt.days / 365.25

    cols = [
        "as_of", "observed_until", "appl_no", "trade_name", "ingredient", "df_route", "status", "main_barrier",
        *[f"pred_{n}" for n in PREDICTORS], "actual_first_generic", "generic_observed", "prior_anda",
        *[f"err_{n}" for n in PREDICTORS],
    ]
    return t[cols].reset_index(drop=True)


MATURITY_YEARS = 2
OUTCOMES = ["Generic >1y early", "Within ±1y", "Generic >1y late", "No generic yet", "Not matured"]


def classify(rows: pd.DataFrame, predictor: str) -> pd.Series:
    """Outcome label per row for one predictor (same rules as summarize)."""
    pred = rows[f"pred_{predictor}"].fillna(rows["as_of"])
    matured = pred <= rows["observed_until"] - pd.DateOffset(years=MATURITY_YEARS)
    e = rows[f"err_{predictor}"]
    label = pd.Series("No generic yet", index=rows.index)
    label[e < -1] = OUTCOMES[0]
    label[e.abs() <= 1] = OUTCOMES[1]
    label[e > 1] = OUTCOMES[2]
    label[~matured] = OUTCOMES[4]
    return label


def summarize(rows: pd.DataFrame, horizon_years: int = HORIZON_YEARS) -> pd.DataFrame:
    """Accuracy per edition and predictor.

    Timing stats use only *matured* predictions: predicted date at least
    MATURITY_YEARS before observed_until, so early, on-time, late and "no generic"
    outcomes are all observable (avoids censoring bias for recent editions).
    Horizon stats ask "will a generic be approved within N years of E?" and are only
    computed when that full window has been observed.
    """
    out = []
    for (as_of, observed_until), g in rows.groupby(["as_of", "observed_until"]):
        clean = g[~g["prior_anda"]]
        horizon_end = as_of + pd.DateOffset(years=horizon_years)
        mature_cut = observed_until - pd.DateOffset(years=MATURITY_YEARS)
        for name in PREDICTORS:
            pred = clean[f"pred_{name}"].fillna(as_of)
            m = clean[pred <= mature_cut]
            e = m[f"err_{name}"]  # NaN where no generic observed
            n = len(m)
            pct = (lambda mask: round(100 * mask.sum() / n, 1) if n else np.nan)
            obs = e.dropna()
            rec = {
                "as_of": as_of.date(),
                "predictor": name,
                "cohort": len(clean),
                "matured": n,
                "early_gt1y_pct": pct(e < -1),
                "within_1y_pct": pct(e.abs() <= 1),
                "late_gt1y_pct": pct(e > 1),
                "no_generic_pct": pct(e.isna()),
                "median_error_yrs": round(obs.median(), 2) if len(obs) else np.nan,
                "mae_yrs": round(obs.abs().mean(), 2) if len(obs) else np.nan,
            }
            if horizon_end <= observed_until:
                pred = clean[f"pred_{name}"].fillna(as_of) <= horizon_end
                act = clean["actual_first_generic"].notna() & (clean["actual_first_generic"] <= horizon_end)
                tp = int((pred & act).sum())
                prec = tp / pred.sum() if pred.sum() else np.nan
                rec_ = tp / act.sum() if act.sum() else np.nan
                rec.update(
                    {
                        f"precision_{horizon_years}y": round(prec, 3),
                        f"recall_{horizon_years}y": round(rec_, 3),
                        f"f1_{horizon_years}y": round(2 * prec * rec_ / (prec + rec_), 3) if prec + rec_ else np.nan,
                    }
                )
            out.append(rec)
    return pd.DataFrame(out)


def run(
    hist_dir: Path = HIST_DIR,
    current_dir: Path = orange_book.DEFAULT_DIR,
    observed_until: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = orange_book.load(current_dir)
    if observed_until is None:
        # The latest ANDA approval in the current file marks how far outcomes are known.
        observed_until = current.products.loc[current.products["appl_type"] == "A", "approval_date"].max()
    rows = []
    for d in sorted(p for p in Path(hist_dir).iterdir() if p.is_dir()):
        as_of = pd.Timestamp(d.name)
        rows.append(backtest_edition(orange_book.load(d), as_of, current, observed_until))
    all_rows = pd.concat(rows, ignore_index=True)
    return all_rows, summarize(all_rows)


def main() -> None:
    rows, summary = run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT_DIR / "backtest_rows.csv", index=False)
    summary.to_csv(OUT_DIR / "backtest_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(summary.to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'backtest_rows.csv'} and backtest_summary.csv")


if __name__ == "__main__":
    main()
