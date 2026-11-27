"""Grounding checks for LLM explanations: is every claim backed by the brief it was given?

Checks:
  * unsupported items: patent numbers, exclusivity codes, NDA numbers, dates, years or percentages
    not in the brief. Numbers and codes are compared in a normal form, so "6,123,456", "US 6123456"
    and "6123456" are one patent, and "EX485", "ex-485" and "EX-485" are one code.
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


# Patent numbers: plain ("6123456", "6123456*PED"), with thousands commas ("6,123,456") or reissues
# ("RE47691"). A "US"/"USP"/"U.S. Patent No." prefix marks a patent even when the number is malformed
# ("USP 8-1441"): its digits are kept so the made-up number is reported.
PATENT_RE = re.compile(
    r"(?<![\w,.])(?P<re>RE\d{5})\b"
    r"|(?<![\w,.$])(?P<comma>\d{1,2},\d{3},\d{3})(?![\d]|,\d)(?P<cped>\*PED)?"
    r"|\b(?P<plain>\d{7,8})(?P<ped>\*PED)?\b"
    r"|\bU\.?S\.?P?\.?\s*(?:[Pp]at(?:ent)?\.?\s*)?(?:[Nn]o\.?\s*)?(?P<us>\d[\d,\-]*\d)\b"
)

# Any code-shaped token, not only real Orange Book formats: small models invent "EX-485", "EX485",
# "ex-485", "U-0000A" or "XA4". Letters, optional hyphen, digits, optional one-letter suffix.
CODE_RE = re.compile(
    r"\b(?!(?:A?NDA|BLA|RE)-?\d)(?P<letters>[A-Za-z]{1,5})(?P<hy>-?)(?P<digits>\d{1,6})(?P<suffix>[A-Za-z]?)\b"
)
WORD_CODES = re.compile(r"\b(?:NPP|NCE|PED|GAIN|ODE)\b")  # uppercase only: "gain" is an English word
# Letter+digit tokens that are not codes: quarters, halves, fiscal years, and ordinary words with a
# number ("type-2", "phase 3" written "phase-3"). "post-2030"-style years are skipped in the code below.
NOT_CODES = re.compile(
    r"(?:Q[1-4]|H[12]|FY\d{2}(?:\d{2})?"
    r"|(?:type|phase|tier|stage|step|grade|class|level|day|week|month|year|top)-?\d{1,3})", re.I)

# Application numbers: "NDA 021234", "NDA021234", "ANDA #076543", "application no. 200677", "BLA 125057".
NDA_RE = re.compile(
    r"\b(?:A?NDA|BLA|application)(?:\s+(?:no\.?|number))?[\s#:\-]*(\d{4,6})\b", re.I)


def norm_code(code: str) -> str:
    return code.upper().replace("-", "")


def patent_numbers(text: str) -> dict[str, str]:
    """Normalized patent number -> how it was written."""
    out = {}
    for m in PATENT_RE.finditer(text):
        if m["re"]:
            n = m["re"].upper()
        elif m["comma"]:
            n = m["comma"].replace(",", "") + ("*PED" if m["cped"] else "")
        elif m["plain"]:
            n = m["plain"] + ("*PED" if m["ped"] else "")
        else:
            n = re.sub(r"\D", "", m["us"])
        out.setdefault(n, m.group().strip())
    return out


def exclusivity_codes(text: str) -> dict[str, str]:
    """Normalized code ("EX485") -> how it was written ("ex-485")."""
    out = {}
    for m in CODE_RE.finditer(text):
        tok = m.group()
        if NOT_CODES.fullmatch(tok):
            continue
        if not m["letters"].isupper() and not m["suffix"] and re.fullmatch(r"(?:19|20)\d\d", m["digits"]):
            continue  # "post-2030", "mid-2033": a year, which the year check covers
        out.setdefault(norm_code(tok), tok)
    for tok in WORD_CODES.findall(text):
        out.setdefault(tok, tok)
    return out


def nda_numbers(text: str) -> dict[str, str]:
    """Application number without leading zeros -> how it was written."""
    return {str(int(m[1])): m.group() for m in NDA_RE.finditer(text)}


def claims(text: str) -> dict[str, set[str]]:
    """Checkable items mentioned in the explanation (patents, codes and NDA numbers normalized)."""
    full_dates = set(re.findall(r"\b(20\d\d|19\d\d)-(\d\d)-(\d\d)\b", text))
    month_dates = re.findall(rf"\b({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", text)
    return {
        "patents": set(patent_numbers(text)),
        "exclusivity_codes": set(exclusivity_codes(text)),
        "nda_numbers": set(nda_numbers(text)),
        "dates": {"-".join(d) for d in full_dates}
        | {pd.Timestamp(f"{m} {d} {y}").strftime("%Y-%m-%d") for m, d, y in month_dates},
        "years": set(re.findall(r"\b(?:19|20)\d\d\b", text)),
        "percentages": {x.replace(" ", "") for x in
                        re.findall(r"\b\d{1,3}(?:\.\d+)?\s?%", text.replace(" percent", "%"))},
    }


def unsupported(text: str, source: str, today: pd.Timestamp, appl_no: str | None = None) -> dict[str, list[str]]:
    """Items in `text` that do not appear in `source` (the brief the model was given).

    Patents, codes and NDA numbers are reported as the explanation wrote them. `appl_no` is the drug's
    own application number, which is allowed even though the brief does not state it."""
    c, src = claims(text), claims(source)
    allowed_years = set(re.findall(r"\b(?:19|20)\d\d\b", source)) | {str(today.year)}
    source_dates = src["dates"]
    # A brief number also covers its pediatric extension ("8000001" allows "8000001*PED") and vice versa.
    src_patents = src["patents"] | {p.split("*")[0] for p in src["patents"]}
    allowed_nda = src["nda_numbers"] | ({str(int(appl_no))} if appl_no and str(appl_no).isdigit() else set())
    pats, codes, ndas = patent_numbers(text), exclusivity_codes(text), nda_numbers(text)
    return {
        "patents": sorted(w for n, w in pats.items() if n not in src_patents and n.split("*")[0] not in src_patents),
        "exclusivity_codes": sorted(w for n, w in codes.items() if n not in src["exclusivity_codes"]),
        "nda_numbers": sorted(w for n, w in ndas.items() if n not in allowed_nda),
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
        for num in patent_numbers(ln):
            num = num.split("*")[0]
            if num in kind and kind[num] not in named:
                bad.append(f"{num} is {kind[num]}, written as {next(iter(named))}")
    return bad


def check(text: str, facts: dict, brief: str, today: pd.Timestamp | None = None) -> dict:
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    bad = unsupported(text, brief, today, facts.get("appl_no"))
    bl = bottom_line_ok(text, facts)
    tm = type_mismatches(text, facts)
    issues = [f"not in brief: {k} {v}" for k, v in bad.items() if v]
    if bl is False:
        issues.append(f"bottom line does not state {facts.get('estimated_entry_all_listed')}")
    issues += tm
    return {"pass": not issues, "issues": issues, "unsupported": bad, "bottom_line_ok": bl, "type_mismatches": tm}
