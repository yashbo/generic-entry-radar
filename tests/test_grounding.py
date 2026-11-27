"""Brief builder and grounding checks (no LLM needed).
Run: python tests/test_grounding.py"""
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


def test_codes_without_hyphen_are_caught():
    text = "* Exclusivity code EX485 and use code U203 apply; XA4 too."
    assert grounding.unsupported(text, BRIEF, TODAY)["exclusivity_codes"] == ["EX485", "U203", "XA4"]


def test_lowercase_codes_are_caught():
    text = "* Exclusivity code ex-485 and use code u-203 apply."
    assert grounding.unsupported(text, BRIEF, TODAY)["exclusivity_codes"] == ["ex-485", "u-203"]


def test_code_spelling_matches_the_brief_in_any_form():
    brief = BRIEF + "\nExclusivity M-14 (new indication) runs until 2029-01-01."
    for text in ("Exclusivity M-14 applies.", "Exclusivity M14 applies.", "Exclusivity m-14 applies."):
        assert grounding.unsupported(text, brief, TODAY)["exclusivity_codes"] == [], text


def test_codes_with_letter_suffix_are_caught():
    text = "* Exclusivity USP-0003A and U-0000A apply."
    assert grounding.unsupported(text, BRIEF, TODAY)["exclusivity_codes"] == ["U-0000A", "USP-0003A"]


def test_words_with_numbers_are_not_codes():
    text = "Entry is post-2030, in Q3 or FY2027, for type-2 patients after phase-3 trials."
    assert grounding.unsupported(text, BRIEF, TODAY)["exclusivity_codes"] == []


def test_patent_numbers_with_commas_are_caught():
    text = "* Patents 6,123,456 and US 10,661,111 and U.S. Patent No. 9,444,111 apply."
    assert grounding.unsupported(text, BRIEF, TODAY)["patents"] == [
        "6,123,456", "U.S. Patent No. 9,444,111", "US 10,661,111"]


def test_patent_numbers_in_the_brief_pass_in_any_form():
    text = "* Patents 9,000,001, US 9000002 and US 9,000,001 apply."
    assert grounding.unsupported(text, BRIEF, TODAY)["patents"] == []


def test_dollar_amounts_are_not_patents():
    assert grounding.unsupported("Spend was $1,234,567 last year.", BRIEF, TODAY)["patents"] == []


def test_malformed_and_reissue_patents_are_caught():
    text = "* USP 8-1441 and reissue RE47691 apply."
    assert grounding.unsupported(text, BRIEF, TODAY)["patents"] == ["RE47691", "USP 8-1441"]


def test_comma_patent_type_mismatch_is_caught():
    text = "* Compound patents (9,000,002) expire 2033-08-15."
    assert grounding.type_mismatches(text, FACTS) and "9000002" in grounding.type_mismatches(text, FACTS)[0]


def test_invented_nda_numbers_are_caught():
    text = "PROTECTA (NDA 021234, see also ANDA #076543 and application no. 200677) is protected."
    assert grounding.unsupported(text, BRIEF, TODAY)["nda_numbers"] == [
        "ANDA #076543", "NDA 021234", "application no. 200677"]
    r = grounding.check("**Bottom line:** Generic entry is expected on 2033-08-15 (NDA 021234).", FACTS, BRIEF, TODAY)
    assert not r["pass"] and r["unsupported"]["nda_numbers"] == ["NDA 021234"]


def test_the_drugs_own_nda_number_is_allowed():
    text = f"**Bottom line:** Generic entry for NDA {FACTS['appl_no']} is expected on 2033-08-15."
    r = grounding.check(text, FACTS, BRIEF, TODAY)
    assert r["pass"], r["issues"]
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
