"""Plain-English explanation of a drug's generic-entry landscape, via a local Ollama model.

Design: code does the analysis, the model only writes. `build_brief` turns the facts into
short labelled sentences (dates already computed, patents already grouped by type), and
the model rewrites that brief for a business reader. Small local models misread raw JSON
(wrong dates, wrong patent types); a pre-digested brief removes that failure mode.
The legal disclaimer is appended in code so it is always present.

Config (env vars):
  OLLAMA_HOST     default http://localhost:11434
  OLLAMA_MODEL    default llama3.2:3b (fits a 4 GB GPU; use llama3.1:8b with more memory)
  OLLAMA_NUM_CTX  default 4096 (briefs are a few hundred tokens)
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Iterator

import requests

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))  # briefs are short; larger spills off a 4 GB GPU
DISCLAIMER = "_Estimate based on listed Orange Book data only; not legal advice._"
NO_EXCLUSIVITY = "Exclusivities: none listed"
MAX_NUMBERS = 3  # patent numbers quoted per barrier type

SYSTEM = """You turn an analyst's brief about one drug into a short explanation for a business reader.

Rules:
- Use only the brief. Copy every date, patent number and exclusivity code exactly as written.
- Never change which type a patent is. Use the type names from the brief.
- Never add facts, litigation outcomes, sales figures or dates that are not in the brief.
- Do not write a disclaimer; one is added automatically.

Format (markdown, under 200 words):
**Bottom line:** one sentence using the brief's "Earliest generic entry" date.
**Barriers:** exactly one bullet per "... patents:" line and per "Exclusivity" line in the brief, in the
brief's order. Never one bullet per patent. Quote that line's patent numbers only in its own bullet.
If the brief says "Exclusivities: none listed", write the bullet "Exclusivity: none listed" and mention no exclusivity codes.
**Carve-out view:** one or two sentences using the brief's carve-out line.
**Watch for:** one or two sentences using the brief's challenge, market and likelihood lines, if present."""

TYPE_MEANING = {
    "Compound patents": "cover the active molecule itself; the hardest barrier to design around",
    "Formulation patents": "cover the product's formulation; a generic can sometimes use a different formulation",
    "Method-of-use patents": "cover specific uses; a generic can often leave those uses off its label (carve-out)",
}


def patent_type(p: dict[str, Any]) -> str:
    if p.get("substance"):
        return "Compound patents"
    if p.get("product"):
        return "Formulation patents"
    return "Method-of-use patents" if p.get("use_codes") else "Other patents"


def patent_types(patents: list[dict[str, Any]]) -> dict[str, str]:
    """Type of each patent number. A pediatric extension ("1234567*PED") carries no type flags in
    the Orange Book, so it takes its base patent's type: it extends that same barrier by 6 months."""
    kind = {p["patent_no"]: patent_type(p) for p in patents}
    for no in kind:
        base = no.split("*")[0]
        if no != base and base in kind:
            kind[no] = kind[base]
    return kind


def build_brief(facts: dict[str, Any]) -> str:
    """Deterministic, human-readable summary that the model rewrites."""
    lines = [
        f"Drug: {facts['trade_name']} ({facts['ingredient']}, {facts['dosage_form_route']}), made by {facts['applicant']}.",
        f"Status: {facts['status']}.",
    ]
    if facts.get("estimated_entry_all_listed"):
        lines.append(
            "Earliest generic entry once every listed patent and exclusivity has expired: "
            f"{facts['estimated_entry_all_listed']} (last barrier: {facts['main_barrier']})."
        )
    else:
        lines.append("Earliest generic entry: no patents or exclusivities are listed.")
    if facts.get("generic_approved"):
        lines.append(
            f"Generics are already FDA-approved ({facts['n_generic_makers']} makers; "
            f"first approval {facts['first_generic_approval']})."
        )

    groups: dict[str, list[dict[str, Any]]] = {}
    kinds = patent_types(facts.get("patents", []))
    for p in facts.get("patents", []):
        if p.get("expires") and not p.get("delisted"):
            groups.setdefault(kinds[p["patent_no"]], []).append(p)
    for kind in sorted(groups, key=lambda k: max(p["expires"] for p in groups[k]), reverse=True):
        ps = sorted(groups[kind], key=lambda p: p["expires"], reverse=True)
        bases = list(dict.fromkeys(p["patent_no"].split("*")[0] for p in ps))  # newest first, extensions folded in
        nums = ", ".join(bases[:MAX_NUMBERS]) + (" and others" if len(bases) > MAX_NUMBERS else "")
        ped = " (dates include 6-month pediatric extensions)" if any("*PED" in p["patent_no"] for p in ps) else ""
        meaning = TYPE_MEANING.get(kind)
        lines.append(
            f"{kind}: {len(bases)} listed, the latest expiring {ps[0]['expires']} ({nums}){ped}"
            + (f"; these {meaning}." if meaning else ".")
        )
    excl = [e for e in facts.get("exclusivities", []) if e.get("expires")]
    for e in excl:
        lines.append(f"Exclusivity {e['exclusivity_code']} ({e['meaning']}) runs until {e['expires']}.")
    if not excl:
        # Said out loud: told only to omit the bullet, small models write one anyway and invent a code.
        lines.append(f"{NO_EXCLUSIVITY}.")

    carve, full = facts.get("estimated_entry_with_carveout"), facts.get("estimated_entry_all_listed")
    if carve and carve != full:
        lines.append(
            "Carve-out view: if method-of-use patents and indication-specific exclusivities are left off "
            f"the generic label, entry could come on {carve}."
        )
    elif carve:
        lines.append(f"Carve-out view: carving out uses does not bring entry earlier than {carve}.")
    elif full:
        lines.append(
            "Carve-out view: compound and formulation protection has already expired, so a generic with "
            "the protected uses carved out of its label could be possible now."
        )

    piv = facts.get("paragraph_iv_challenge")
    if piv:
        bits = [f"first filed {piv['first_filed']}"]
        if piv.get("number_of_challenging_generics"):
            bits.append(f"{piv['number_of_challenging_generics']} challenger(s)")
        if piv.get("180_day_exclusivity_status"):
            bits.append(f"180-day exclusivity status: {piv['180_day_exclusivity_status']}")
        if piv.get("first_challenger_launched"):
            bits.append(f"first challenger launched {piv['first_challenger_launched']}")
        lines.append(
            "Patent challenge (Paragraph IV): generic makers are challenging the patents, so entry can come "
            "before the listed dates through court wins or settlements; " + "; ".join(bits) + "."
        )
    for k, v in facts.items():
        if k.startswith("medicare_part_d_brand_spend_"):
            year = k.split("_")[-3]
            lines.append(
                f"Market: Medicare Part D spent ${v:,.0f} million on this brand in {year} "
                "(Medicare retail prescriptions only)."
            )
    if facts.get("model_generic_likelihood_pct") is not None:
        lines.append(
            f"Statistical model: {facts['model_generic_likelihood_pct']}% chance a generic is approved within "
            "2 years of protection ending (whether generic makers will bother at all)."
        )
    return "\n".join(lines)


def available() -> tuple[bool, str]:
    """Return (ok, message). ok is True when the server is up and the model is pulled."""
    try:
        r = requests.get(f"{HOST}/api/tags", timeout=3)
        r.raise_for_status()
    except requests.RequestException:
        return False, f"Ollama is not running at {HOST}. Install it from ollama.com, then run `ollama serve`."
    names = {m.get("name", "") for m in r.json().get("models", [])}
    if MODEL not in names and f"{MODEL}:latest" not in names:
        return False, f"Model `{MODEL}` is not pulled. Run `ollama pull {MODEL}` (or set OLLAMA_MODEL)."
    return True, f"Using {MODEL}"


def _messages(facts: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Brief:\n" + build_brief(facts)}]


# Hard limits. Without num_predict a small model can fall into a repetition loop and
# Ollama keeps generating forever (it shifts the context window instead of stopping),
# so the HTTP read timeout never fires because tokens keep arriving.
MAX_TOKENS = 450          # the format asks for <200 words (~300 tokens)
MAX_SECONDS = 90          # wall-clock cap per answer


class GenerationTimeout(RuntimeError):
    pass


def _post(facts: dict[str, Any]) -> requests.Response:
    body = {"model": MODEL, "messages": _messages(facts), "stream": True,
            "options": {"temperature": 0.1, "num_ctx": NUM_CTX, "num_predict": MAX_TOKENS,
                        "repeat_penalty": 1.15}}
    for attempt in range(2):
        r = requests.post(f"{HOST}/api/chat", json=body, stream=True, timeout=300)
        # A 5xx usually means the model runner crashed while loading; Ollama restarts it.
        if r.status_code < 500 or attempt == 1:
            r.raise_for_status()
            return r
        r.close()
        time.sleep(3)
    raise RuntimeError("unreachable")


def explain_stream(facts: dict[str, Any]) -> Iterator[str]:
    """Yield the explanation text as it is generated, then the disclaimer."""
    start = time.monotonic()
    with _post(facts) as r:
        for line in r.iter_lines():
            if time.monotonic() - start > MAX_SECONDS:
                # Closing the response makes Ollama abort the generation.
                raise GenerationTimeout(f"stopped after {MAX_SECONDS}s")
            if not line:
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
            if chunk.get("done"):
                break
    yield "\n\n" + DISCLAIMER


def explain(facts: dict[str, Any]) -> str:
    return "".join(explain_stream(facts))


def brief_as_markdown(facts: dict[str, Any]) -> str:
    """Fallback shown when the model's text fails the grounding checks."""
    return "\n".join(f"- {ln}" for ln in build_brief(facts).splitlines()) + "\n\n" + DISCLAIMER


def explain_checked(facts: dict[str, Any], attempts: int = 2) -> tuple[str, dict]:
    """Generate, verify against the brief, retry once, else fall back to the brief itself.

    Returns (markdown, report) where report has pass/issues/attempts/fallback/history.
    """
    import grounding

    brief = build_brief(facts)
    report: dict = {}
    history: list[dict] = []  # every attempt's issues and text, so a failure can be diagnosed later
    for i in range(1, attempts + 1):
        try:
            text = explain(facts)
        except GenerationTimeout as exc:
            report = {"pass": False, "issues": [f"generation {exc}"]}
            text = ""
        else:
            report = grounding.check(text, facts, brief)
        history.append({"attempt": i, "issues": report["issues"], "text": text})
        report.update(attempts=i, fallback=False, history=history)
        if report["pass"]:
            return text, report
    report["fallback"] = True
    return brief_as_markdown(facts), report
