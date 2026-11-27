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


def test_pediatric_extension_takes_its_base_patent_type():
    # Real Orange Book *PED rows have no substance/product flags; they extend the base patent.
    facts = {**FACTS, "patents": [
        {"patent_no": "8000001", "expires": "2040-07-06", "product": True},
        {"patent_no": "8000001*PED", "expires": "2041-01-06"},
    ]}
    brief = explain.build_brief(facts)
    assert "Other patents" not in brief
    assert ("Formulation patents: 1 listed, the latest expiring 2041-01-06 (8000001) "
            "(dates include 6-month pediatric extensions)") in brief


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


def test_invented_code_formats_are_caught():
    text = "* Exclusivity code EX-485 and use code U-203 apply."
    assert grounding.unsupported(text, BRIEF, TODAY)["exclusivity_codes"] == ["EX-485", "U-203"]


def test_brief_says_when_no_exclusivity_is_listed():
    facts = {**FACTS, "exclusivities": []}
    assert explain.NO_EXCLUSIVITY in explain.build_brief(facts)


def test_invented_exclusivity_is_caught_when_brief_has_none():
    brief = explain.build_brief({**FACTS, "exclusivities": []})
    for line in ["* Exclusivity code: XA4T8J3RXC", "• Exclusivity USP-0003A",
                 "* Orphan drug exclusivity runs until 2031."]:
        assert grounding.invented_exclusivity(line, brief) == [line], line
    for line in ["* Exclusivity:", "* Exclusivity: none listed", "* Orphan drug exclusivity: Not applicable.",
                 "**Carve-out view:** leaving off indication-specific exclusivities does not help."]:
        assert grounding.invented_exclusivity(line, brief) == [], line


def test_exclusivity_bullets_are_fine_when_the_brief_lists_one():
    brief = BRIEF + "\nExclusivity ODE-428 (Orphan drug exclusivity) runs until 2030-03-16."
    assert grounding.invented_exclusivity("* Exclusivity ODE-428 runs until 2030-03-16.", brief) == []


def test_code_with_trailing_letters_is_caught():
    assert grounding.unsupported("* Exclusivity USP-0003A", BRIEF, TODAY)["exclusivity_codes"] == ["USP-0003A"]


def test_percentages_must_match_the_brief():
    brief = BRIEF + "\nStatistical model: 62% chance a generic is approved within 2 years of protection ending."
    assert grounding.unsupported("The model gives a 62 percent chance.", brief, TODAY)["percentages"] == []
    assert grounding.unsupported("The model gives a 61 % chance.", brief, TODAY)["percentages"] == ["61%"]


def test_month_name_dates_are_understood():
    text = "**Bottom line:** Generic entry is expected on August 15, 2033."
    assert grounding.bottom_line_ok(text, FACTS) is True
    assert grounding.bottom_line_ok("**Bottom line:** Generic entry is expected on August 15th, 2033.", FACTS) is True
    assert grounding.bottom_line_ok("**Bottom line:** Generic entry is expected on March 1st, 2031.", FACTS) is False


def test_month_and_year_only_bottom_line():
    # Entry is 2033-08-15: "August 2033" is right, "March 2031" (the carve-out month) is wrong.
    assert grounding.bottom_line_ok("**Bottom line:** Generic entry is expected in August 2033.", FACTS) is True
    assert grounding.bottom_line_ok("**Bottom line:** Generic entry is expected in March 2031.", FACTS) is False
    assert grounding.bottom_line_ok("**Bottom line:** Generic entry is expected soon.", FACTS) is False


def test_every_failed_attempt_is_recorded(monkeypatch=None):
    answers = iter(["**Bottom line:** entry on 2031-03-01.", "* Exclusivity code EX-485 applies."])
    real = explain.explain
    explain.explain = lambda facts: next(answers)
    try:
        md, report = explain.explain_checked(FACTS)
    finally:
        explain.explain = real
    assert report["fallback"] and [h["attempt"] for h in report["history"]] == [1, 2]
    assert any("bottom line" in i for i in report["history"][0]["issues"])
    assert any("EX-485" in i for i in report["history"][1]["issues"])
    assert report["history"][1]["text"].startswith("* Exclusivity")


def test_fallback_is_the_brief_plus_disclaimer():
    md = explain.brief_as_markdown(FACTS)
    assert md.startswith("- Drug: PROTECTA") and md.rstrip().endswith(explain.DISCLAIMER)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("pass", name)
