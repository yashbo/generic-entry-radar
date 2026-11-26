"""Stored AI explanations are only shown while they still check out against today's facts.
Run: python generic_entry_radar/tests/test_explanations.py"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import entry  # noqa: E402
import explain  # noqa: E402
import explanations  # noqa: E402
import orange_book  # noqa: E402

TODAY = pd.Timestamp("2026-11-26")
OB = orange_book.load(ROOT / "tests" / "fixtures" / "orange_book")
T = entry.build_entry_table(OB, today=TODAY)
FACTS = entry.drug_facts(OB, T, "000001")  # PROTECTA: entry 2033-08-15, carve-out 2031-03-01
TEXT = ("**Bottom line:** Generic entry is expected on 2033-08-15.\n"
        "* Method-of-use patents (9000002) run to 2033-08-15.\n"
        "* Compound patents (9000001) run to 2031-03-01.\n\n" + explain.DISCLAIMER)
STORE = {"000001": {"text": TEXT, "brief": explain.build_brief(FACTS), "model": "test", "generated": "2026-11-26"}}


def test_shown_while_facts_unchanged():
    e, report = explanations.current(STORE, FACTS, TODAY)
    assert e is not None and report["pass"]


def test_hidden_when_a_date_changes():
    # A patent term extension moves the entry date: the stored bottom line is now wrong.
    later = {**FACTS, "estimated_entry_all_listed": "2034-02-15",
             "patents": [dict(p, expires="2034-02-15") if p["patent_no"] == "9000002" else p for p in FACTS["patents"]]}
    e, report = explanations.current(STORE, later, TODAY)
    assert e is None and not report["pass"]


def test_hidden_when_status_changes():
    e, report = explanations.current(STORE, {**FACTS, "status": "Carve-out possible now"}, TODAY)
    assert e is None and "status changed since the answer was written" in report["issues"]


def test_no_answer_stored():
    assert explanations.current(STORE, {**FACTS, "appl_no": "999999"}, TODAY) == (None, None)


def test_pick_puts_soonest_protection_end_first():
    ob, today = OB, TODAY
    table = T.assign(partd_spend_m=0.0, piv_first_date=pd.NaT)
    picks = explanations.pick(table, today, limit=50)
    prot = table[table["status"] == "Protected"]
    soon = prot[prot["entry_all_listed"] <= today + pd.DateOffset(years=5)].sort_values("entry_all_listed")
    assert picks[:len(soon)] == soon["appl_no"].tolist()
    assert len(picks) == len(set(picks))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("pass", name)
