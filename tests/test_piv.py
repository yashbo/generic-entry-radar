"""Paragraph IV list parsing and point-in-time features, on a generated mini spreadsheet.
Run: python tests/test_piv.py"""
import datetime as dt
import sys
import tempfile
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import piv  # noqa: E402

HEADER = ["DRUG NAME", "DOSAGE FORM", "STRENGTH", "RLD/NDA", "DATE OF SUBMISSION", "NUMBER OF ANDAs SUBMITTED",
          "180-DAY STATUS", "180-DAY DECISION POSTING DATE", "DATE OF FIRST APPLICANTAPPROVAL",
          "DATE OF FIRST COMMERCIAL MARKETING BY FTF", "EXPIRATION DATE OF LAST QUALIFYING PATENT"]
ROWS = [
    # Two strengths of one NDA: earliest date and largest ANDA count must win.
    ["Omegastat", "Tablets", "40 mg", "Blocked\n7", "6/1/2019", 2, "Eligible", None, dt.datetime(2025, 1, 1), None, dt.datetime(2030, 6, 1)],
    ["Omegastat", "Tablets", "80 mg", "Blocked\n7", dt.datetime(2021, 3, 1), 5, None, None, None, None, None],
    ["Alphanib", "Tablets", "10 mg", "Protecta\n000001", "Pre-MMA", 1, "Extinguished", None, None, None, None],
    ["Oldbrand", "Capsules", "5 mg", "Oldbrand", "1/2/2008", 1, None, None, None, None, None],  # no NDA number
]


def _make(tmp: Path) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["10/27/2026 Paragraph IV List Updates (Public)"])
    ws.append(HEADER)
    for r in ROWS:
        ws.append(r)
    path = tmp / "piv.xlsx"
    wb.save(path)
    return path


TMP = Path(tempfile.mkdtemp())
RAW = piv.load_rows(_make(TMP))
P = piv.by_nda(RAW).set_index("appl_no")


def test_parses_mixed_dates_and_pre_mma():
    assert RAW.loc[0, "piv_date"] == pd.Timestamp("2019-06-01")
    assert RAW.loc[1, "piv_date"] == pd.Timestamp("2021-03-01")
    assert RAW.loc[2, "piv_date"] == piv.PRE_MMA and RAW.loc[2, "piv_date_approx"]


def test_nda_numbers_normalised_and_missing_ones_skipped():
    # "Blocked\n7" has no 5-6 digit number, so is skipped like "Oldbrand": only NDA 000001 parses.
    assert set(P.index) == {"000001"}


def test_aggregates_per_nda():
    rows = RAW.copy()
    rows.loc[[0, 1], "appl_nos"] = pd.Series([["000007"], ["000007"]], index=[0, 1])
    agg = piv.by_nda(rows).set_index("appl_no").loc["000007"]
    assert agg["piv_first_date"] == pd.Timestamp("2019-06-01")
    assert agg["piv_max_andas"] == 5
    assert agg["piv_ftf_approval"] == pd.Timestamp("2025-01-01")


def test_features_only_count_challenges_filed_before_as_of():
    table = pd.DataFrame({"appl_no": ["000007", "000001", "000003"]})
    p = pd.DataFrame({"appl_no": ["000007", "000001"],
                      "piv_first_date": [pd.Timestamp("2019-06-01"), piv.PRE_MMA]})
    f = piv.piv_features(table, p, pd.Timestamp("2018-01-01")).set_index("appl_no")
    assert f.loc["000007", "piv_filed"] == 0 and f.loc["000007", "years_since_piv"] == 0  # filed after as_of
    assert f.loc["000001", "piv_filed"] == 1 and f.loc["000001", "years_since_piv"] > 14
    assert f.loc["000003", "piv_filed"] == 0
    f2 = piv.piv_features(table, p, pd.Timestamp("2020-06-01")).set_index("appl_no")
    assert f2.loc["000007", "piv_filed"] == 1


def test_no_list_means_no_challenges():
    f = piv.piv_features(pd.DataFrame({"appl_no": ["000001"]}), None, pd.Timestamp("2020-01-01"))
    assert f["piv_filed"].tolist() == [0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("pass", name)
