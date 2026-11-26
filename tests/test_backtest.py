"""Backtest tests: a synthetic old-format 2020 edition scored against the main fixture.
Run: python generic_entry_radar/tests/test_backtest.py"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backtest  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
UNTIL = pd.Timestamp("2026-11-26")
ROWS, SUMMARY = backtest.run(FIX / "history", FIX / "orange_book", observed_until=UNTIL)
R = ROWS.set_index("appl_no")


def test_old_format_loads_and_cohort_is_drugs_without_generics():
    # 000003 is in the cohort too: its patent ran to 2021, so it was protected in 2020.
    assert set(R.index) == {"000001", "000003", "000007"}


def test_actual_generic_matched_across_route_spacing():
    # Edition says "TABLET; ORAL", current says "TABLET;ORAL": must still match.
    r = R.loc["000007"]
    assert r["actual_first_generic"] == pd.Timestamp("2025-01-01")
    assert round(r["err_all_listed"], 2) == round((pd.Timestamp("2025-01-01") - pd.Timestamp("2024-06-01")).days / 365.25, 2)


def test_no_generic_is_nan_not_error():
    assert pd.isna(R.loc["000001", "actual_first_generic"])
    assert pd.isna(R.loc["000001", "err_all_listed"])


def test_summary_only_scores_matured_predictions():
    s = SUMMARY[SUMMARY["predictor"] == "all_listed"].iloc[0]
    # 000001 (2031) is not matured; 000007 (2024-06) and 000003 (2021) are.
    assert s["matured"] == 2
    assert s["within_1y_pct"] == 50.0  # 000007 within a year
    assert s["no_generic_pct"] == 50.0  # 000003 never got a generic
    # A 3-year horizon from 2020-01-01 is fully observed by 2026-11-26.
    assert "precision_3y" in SUMMARY.columns


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("pass", name)
