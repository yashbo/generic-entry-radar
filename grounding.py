"""Grounding checks for LLM explanations: is every claim backed by the brief it was given?

Checks:
  * unsupported items: patent numbers, exclusivity codes, dates, years or percentages not in the brief
  * bottom line: its date must be the true all-listed entry date
  * patent types: a patent number must not sit on a line naming a different patent type
  * invented exclusivity: when the brief lists none, the text must not describe one
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
    month_dates = re.findall(rf"\b({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", text)
    return {
        "patents": set(re.findall(r"\b\d{7,8}(?:\*PED)?\b", text)),
        # Any code-shaped token, not only real Orange Book formats: small models invent "EX-485" or "U-203".
        "exclusivity_codes": set(re.findall(r"\b(?:[A-Z]{1,5}-\d+[A-Z0-9]*|NPP|NCE|PED|GAIN|ODE)\b", text)),
        "dates": {"-".join(d) for d in full_dates}
        | {pd.Timestamp(f"{m} {d} {y}").strftime("%Y-%m-%d") for m, d, y in month_dates},
        "years": set(re.findall(r"\b(?:19|20)\d\d\b", text)),
        "percentages": {x.replace(" ", "") for x in
                        re.findall(r"\b\d{1,3}(?:\.\d+)?\s?%", text.replace(" percent", "%"))},
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
        "percentages": sorted(x for x in c["percentages"] if x not in claims(source)["percentages"]),
    }


def bottom_line_ok(text: str, facts: dict) -> bool | None:
    """The Bottom line must state the all-listed entry date: the exact day, or its month and year."""
    want = facts.get("estimated_entry_all_listed")
    if not want:
        return None
    line = next((ln for ln in text.splitlines() if "bottom line" in ln.lower()), "")
    got = claims(line)["dates"]
    if got:
        return got == {want}
    # Month-and-year only ("May 2037"): accept if it is the right month, reject any other month.
    months = {pd.Timestamp(f"1 {m} {y}").strftime("%Y-%m")
              for m, y in re.findall(rf"\b({MONTHS})\s+(\d{{4}})\b", line)}
    return months == {want[:7]}


def type_mismatches(text: str, facts: dict) -> list[str]:
    """Patent numbers placed on a line that names a different patent type."""
    from explain import patent_types
    kind = patent_types(facts.get("patents", []))
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


NEGATION = re.compile(r"\b(?:none|no|not|n/a)\b", re.I)


def invented_exclusivity(text: str, brief: str) -> list[str]:
    """Lines describing an exclusivity when the brief lists none.

    Small models ignore "write no exclusivity bullet" and fill one in ("Exclusivity code: XA4T8J3RXC",
    "Orphan drug exclusivity ... until 2031"). Only code-shaped tokens were caught before, so most
    slipped through. Allowed: empty or negative bullets; the bottom line, carve-out and watch-for
    sentences, where "exclusivity" is loose prose for protection; and the brief's own
    180-day (Paragraph IV) exclusivity wording."""
    if re.search(r"^Exclusivity ", brief, re.M):
        return []
    bad = []
    for ln in text.splitlines():
        low = ln.lower()
        if "exclusivit" not in low or any(s in low for s in ("carve-out", "bottom line", "watch for")):
            continue
        if "180-day" in low and "180-day exclusivity" in brief:
            continue
        rest = re.sub(r"[*•+#_\s:.-]|exclusivit(?:y|ies)", "", low)
        if not rest or NEGATION.search(low):
            continue
        bad.append(ln.strip())
    return bad


def check(text: str, facts: dict, brief: str, today: pd.Timestamp | None = None) -> dict:
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    bad = unsupported(text, brief, today)
    bl = bottom_line_ok(text, facts)
    tm = type_mismatches(text, facts)
    ie = invented_exclusivity(text, brief)
    issues = [f"not in brief: {k} {v}" for k, v in bad.items() if v]
    if bl is False:
        issues.append(f"bottom line does not state {facts.get('estimated_entry_all_listed')}")
    issues += tm
    issues += [f"exclusivity not in brief: {ln[:80]}" for ln in ie]
    return {"pass": not issues, "issues": issues, "unsupported": bad, "bottom_line_ok": bl,
            "type_mismatches": tm, "invented_exclusivity": ie}
