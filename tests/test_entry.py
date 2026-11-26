"""Tests on a synthetic Orange Book fixture. Run: python -m pytest generic_entry_radar/tests
or simply: python generic_entry_radar/tests/test_entry.py"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import entry  # noqa: E402
import orange_book  # noqa: E402

TODAY = pd.Timestamp("2026-11-26")
OB = orange_book.load(ROOT / "tests" / "fixtures" / "orange_book")
T = entry.build_entry_table(OB, today=TODAY).set_index("appl_no")


def test_protected_by_compound_and_use_patents():
    r = T.loc["000001"]
    assert r["status"] == "Protected"
    assert r["main_barrier"] == "Method-of-use patent"
    assert r["entry_all_listed"] == pd.Timestamp("2033-08-15")
    assert r["entry_with_carveout"] == pd.Timestamp("2031-03-01")
    assert r["n_strengths"] == 2 and r["n_patents"] == 2


def test_generic_detected_via_anda_match():
    r = T.loc["000002"]
    assert r["status"] == "Generic approved"
    assert r["n_generic_makers"] == 2
    assert r["first_generic_approval"] == pd.Timestamp("2010-06-01")
    assert r["years_to_entry"] == 0


def test_open_opportunity_when_protection_expired():
    assert T.loc["000003", "status"] == "Open: no listed barrier, no generic"


def test_carveout_ignores_use_patents_and_indication_exclusivity_and_delisted():
    r = T.loc["000004"]
    # The 2035 formulation patent is delisted, so it must not count.
    assert r["entry_all_listed"] == pd.Timestamp("2029-09-09")
    assert r["status"] == "Carve-out possible now"


def test_exclusivity_and_pediatric_extension():
    r = T.loc["000005"]
    assert r["main_barrier"] == "Exclusivity"
    assert r["entry_all_listed"] == pd.Timestamp("2028-07-01")
    assert r["n_patents"] == 1  # *PED row folds into its base patent


def test_generic_approved_while_patents_listed_is_flagged():
    r = T.loc["000007"]
    assert r["status"] == "Generic approved, patents still listed"
    assert r["years_to_entry"] > 0


def test_npp_is_carveoutable():
    assert entry.describe_exclusivity("NPP").startswith("New patient population")
    assert entry._is_carveout_excl(pd.Series(["NPP", "NCE"])).tolist() == [True, False]


def test_discontinued_ndas_excluded():
    assert "000006" not in T.index


def test_drug_facts_shape():
    f = entry.drug_facts(OB, T.reset_index(), "000005")
    assert f["trade_name"] == "NEWBIE"
    assert {p["patent_no"] for p in f["patents"]} == {"8000001", "8000001*PED"}
    assert f["exclusivities"][0]["meaning"].startswith("New chemical entity")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("pass", name)
