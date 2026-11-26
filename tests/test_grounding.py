"""Brief builder and grounding checks (no LLM needed).
Run: python generic_entry_radar/tests/test_grounding.py"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import entry  # noqa: E402
import explain  # noqa: E402
import grounding  # noqa: E402
import orange_book  # noqa: E402

TODAY = pd.Timestamp("2026-11-26")
OB = orange_book.load(ROOT / "tests" / "fixtures" / "orange_book")
T = entry.build_entry_table(OB, today=TODAY)
FACTS = entry.drug_facts(OB, T, "000001")  # PROTECTA: compound 2031, method-of-use 2033
BRIEF = explain.build_brief(FACTS)


def test_brief_has_computed_dates_and_grouped_types():
    assert "2033-08-15 (last barrier: Method-of-use patent)" in BRIEF
    assert "Compound patents: 1 listed, the latest expiring 2031-03-01 (9000001)" in BRIEF
    assert "Method-of-use patents: 1 listed, the latest expiring 2033-08-15 (9000002)" in BRIEF
    assert "entry could come on 2031-03-01" in BRIEF  # carve-out date


def test_brief_skips_delisted_patents():
    facts = entry.drug_facts(OB, T, "000004")  # 7000003 is delisted
    assert "7000003" not in explain.build_brief(facts)


def test_good_explanation_passes():
    text = ("**Bottom line:** Generic entry is expected on 2033-08-15.\n"
            "* Method-of-use patents (9000002) run to 2033-08-15.\n"
            "* Compound patents (9000001) run to 2031-03-01.\n"
            "**Carve-out view:** entry could come on 2031-03-01.")
    r = grounding.check(text, FACTS, BRIEF, TODAY)
    assert r["pass"], r["issues"]


def test_wrong_bottom_line_date_fails():
    text = "**Bottom line:** Generic entry is expected on 2031-03-01."  # carve-out date, not the entry date
    r = grounding.check(text, FACTS, BRIEF, TODAY)
    assert not r["pass"] and r["bottom_line_ok"] is False


def test_invented_patent_and_wrong_type_fail():
    text = ("**Bottom line:** Generic entry is expected on 2033-08-15.\n"
            "* Compound patents (9000002) and patent 1234567 expire 2033-08-15.")
    r = grounding.check(text, FACTS, BRIEF, TODAY)
    assert r["unsupported"]["patents"] == ["1234567"]
    assert r["type_mismatches"] and "9000002" in r["type_mismatches"][0]


def test_month_name_dates_are_understood():
    text = "**Bottom line:** Generic entry is expected on August 15, 2033."
    assert grounding.bottom_line_ok(text, FACTS) is True


def test_fallback_is_the_brief_plus_disclaimer():
    md = explain.brief_as_markdown(FACTS)
    assert md.startswith("- Drug: PROTECTA") and md.rstrip().endswith(explain.DISCLAIMER)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("pass", name)
