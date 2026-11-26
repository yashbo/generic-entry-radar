"""Pre-written, fact-checked AI explanations for the hosted app (which has no LLM).

Generate locally with Ollama:
    python generic_entry_radar/explanations.py --limit 300

Each drug goes through the same pipeline as the app's Explain button (write, fact-check against
the brief, one retry). Only answers that pass are stored, together with the brief they were
checked against. At display time the app checks the stored answer again against today's data
(`current`), so an answer whose dates or numbers have gone stale is not shown.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import explain  # noqa: E402
import grounding  # noqa: E402

STORE = ROOT / "data" / "explanations.json"


def load(path: Path = STORE) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("explanations", {})


def current(store: dict[str, Any], facts: dict[str, Any], today: pd.Timestamp | None = None):
    """The stored answer for this drug if it still passes the fact check on today's facts, else None.

    Returns (entry, report); entry is None when there is no answer or it no longer checks out."""
    e = store.get(facts["appl_no"])
    if not e:
        return None, None
    brief = explain.build_brief(facts)
    report = grounding.check(e["text"], facts, brief, today)
    status_now = next((ln for ln in brief.splitlines() if ln.startswith("Status:")), "")
    if status_now not in e["brief"]:  # e.g. protection has since ended: the prose would be out of date
        report["pass"] = False
        report["issues"].append("status changed since the answer was written")
    return (e if report["pass"] else None), report


def pick(table: pd.DataFrame, today: pd.Timestamp, limit: int) -> list[str]:
    """Most useful drugs first: protection ending soon, carve-out now, big Medicare spend, PIV challenge."""
    prot = table[table["status"] == "Protected"]
    soon = prot[prot["entry_all_listed"] <= today + pd.DateOffset(years=5)].sort_values("entry_all_listed")
    carve = table[table["status"] == "Carve-out possible now"].sort_values("partd_spend_m", ascending=False)
    spend = prot.sort_values("partd_spend_m", ascending=False)
    spend = spend[spend["partd_spend_m"] > 0]
    challenged = prot[prot["piv_first_date"].notna()].sort_values("piv_first_date")
    order = pd.concat([soon, carve, spend, challenged])["appl_no"]
    return list(dict.fromkeys(order))[:limit]


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--redo", action="store_true", help="Regenerate drugs that already have a stored answer")
    args = ap.parse_args()

    import orange_book
    import radar

    ok, msg = explain.available()
    if not ok:
        print(msg)
        return 1
    today = pd.Timestamp.today().normalize()
    ob, table = radar.build_table(orange_book.DEFAULT_DIR, today)
    store = load()
    picks = pick(table, today, args.limit)
    stats = {"tried": 0, "passed": 0, "failed": 0, "kept": 0}
    t_start = time.time()
    for i, appl_no in enumerate(picks, 1):
        facts = radar.full_facts(ob, table, appl_no)
        if not args.redo and current(store, facts, today)[0]:
            stats["kept"] += 1
            continue
        stats["tried"] += 1
        text, report = explain.explain_checked(facts)
        status = "PASS" if not report["fallback"] else "FAIL " + "; ".join(report["issues"][:2])
        print(f"[{i}/{len(picks)}] {facts['trade_name'][:30]:30s} {status}", flush=True)
        if report["fallback"]:
            stats["failed"] += 1
            store.pop(appl_no, None)  # never keep an answer that no longer matches
            continue
        stats["passed"] += 1
        store[appl_no] = {"trade_name": facts["trade_name"], "text": text, "brief": explain.build_brief(facts),
                          "model": explain.MODEL, "generated": today.strftime("%Y-%m-%d"),
                          "attempts": report["attempts"]}
        if i % 10 == 0:
            _save(store, stats)
    _save(store, stats)
    print(f"\n{stats}  in {time.time() - t_start:.0f}s; {len(store)} answers stored in {STORE}")
    return 0


def _save(store: dict[str, Any], stats: dict[str, int]) -> None:
    STORE.write_text(json.dumps({"model": explain.MODEL, "stats": stats, "explanations": store},
                                indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
