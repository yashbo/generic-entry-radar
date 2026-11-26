"""Grounding checks for LLM explanations: is every claim backed by the brief it was given?

Checks:
  * unsupported items: patent numbers, exclusivity codes, dates or years not in the brief
  * bottom line: its date must be the true all-listed entry date
  * patent types: a patent number must not sit on a line naming a different patent type
"""
from __future__ import annotations

import re

import pandas as pd

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


TYPE_WORDS = {
    "Compound patents": ("compound", "substance", "active molecule"),
    "Formulation patents": ("formulation",),
    "Method-of-use patents": ("method-of-use", "method of use", "use patent"),
}


def claims(text: str) -> dict[str, set[str]]:
    """Checkable items mentioned in the explanation."""
    full_dates = set(re.findall(r"\b(20\d\d|19\d\d)-(\d\d)-(\d\d)\b", text))
    month_dates = re.findall(rf"\b({MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", text)
    return {
        "patents": set(re.findall(r"\b\d{7,8}(?:\*PED)?\b", text)),
        "exclusivity_codes": set(re.findall(r"\b(?:ODE-\d+|I-\d+|D-\d+|M-\d+|NPP|NCE|PED|GAIN)\b", text)),
        "dates": {"-".join(d) for d in full_dates}
        | {pd.Timestamp(f"{m} {d} {y}").strftime("%Y-%m-%d") for m, d, y in month_dates},
        "years": set(re.findall(r"\b(?:19|20)\d\d\b", text)),
    }


def unsupported(text: str, source: str, today: pd.Timestamp) -> dict[str, list[str]]:
    """Items in `text` that do not appear in `source` (the brief the model was given)."""
    c = claims(text)
    allowed_years = set(re.findall(r"\b(?:19|20)\d\d\b", source)) | {str(today.year)}
    source_dates = claims(source)["dates"]
    return {
        "patents": sorted(p for p in c["patents"] if p not in source),
        "exclusivity_codes": sorted(e for e in c["exclusivity_codes"] if e not in source),
        "dates": sorted(d for d in c["dates"] if d not in source_dates),
        "years": sorted(y for y in c["years"] if y not in allowed_years),
    }


def bottom_line_ok(text: str, facts: dict) -> bool | None:
    """The first date in the Bottom line must be the all-listed entry date."""
    want = facts.get("estimated_entry_all_listed")
    line = next((ln for ln in text.splitlines() if "bottom line" in ln.lower()), "")
    got = sorted(claims(line)["dates"])
    if not want:
        return None
    return bool(got) and want in got and len(got) == 1


def type_mismatches(text: str, facts: dict) -> list[str]:
    """Patent numbers placed on a line that names a different patent type."""
    from explain import patent_type
    kind = {p["patent_no"]: patent_type(p) for p in facts.get("patents", [])}
    bad = []
    for ln in text.splitlines():
        low = ln.lower()
        named = {k for k, words in TYPE_WORDS.items() if any(w in low for w in words)}
        if len(named) != 1:
            continue  # no type, or several types discussed on one line: ambiguous
        for num in re.findall(r"\b\d{7,8}\b", ln):
            if num in kind and kind[num] not in named:
                bad.append(f"{num} is {kind[num]}, written as {next(iter(named))}")
    return bad


def check(text: str, facts: dict, brief: str, today: pd.Timestamp | None = None) -> dict:
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    bad = unsupported(text, brief, today)
    bl = bottom_line_ok(text, facts)
    tm = type_mismatches(text, facts)
    issues = [f"not in brief: {k} {v}" for k, v in bad.items() if v]
    if bl is False:
        issues.append(f"bottom line does not state {facts.get('estimated_entry_all_listed')}")
    issues += tm
    return {"pass": not issues, "issues": issues, "unsupported": bad, "bottom_line_ok": bl, "type_mismatches": tm}
