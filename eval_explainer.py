"""Grounding check for the LLM explainer.

For a sample of drugs, generate an explanation and flag every patent number,
exclusivity code, full date or year that does NOT appear in the facts the model was given.
Unsupported items are likely hallucinations.

Run: python generic_entry_radar/eval_explainer.py [--n 8] [--model llama3.2:3b]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "explainer_eval.json"

from grounding import bottom_line_ok, claims, type_mismatches, unsupported  # noqa: E402,F401


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--model", default=None, help="Override OLLAMA_MODEL")
    args = ap.parse_args()
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model
    sys.path.insert(0, str(ROOT))
    import entry
    import explain
    import orange_book
    import piv

    ok, msg = explain.available()
    if not ok:
        print(msg)
        return 1
    today = pd.Timestamp.today().normalize()
    ob = orange_book.load()
    table = entry.build_entry_table(ob, today=today)
    pv = piv.load()
    if pv is not None:
        table = table.merge(pv, on="appl_no", how="left")

    # A mix of situations: protected with/without a challenge, carve-out, generics blocked by patents.
    picks = []
    for status, n in [("Protected", args.n // 2), ("Carve-out possible now", 1),
                      ("Generic approved, patents still listed", args.n - args.n // 2 - 1)]:
        pool = table[(table["status"] == status) & (table["n_patents"] > 0)]
        picks += pool.sample(min(n, len(pool)), random_state=1)["appl_no"].tolist()

    results = []
    for appl_no in picks:
        facts = entry.drug_facts(ob, table, appl_no)
        row = table.loc[table["appl_no"] == appl_no].iloc[0]
        if pv is not None and pd.notna(row.get("piv_first_date")):
            facts["paragraph_iv_challenge"] = {"first_filed": row["piv_first_date"].strftime("%Y-%m-%d"),
                                               "number_of_challenging_generics": None if pd.isna(row["piv_max_andas"]) else int(row["piv_max_andas"])}
        brief = explain.build_brief(facts)
        t0 = time.time()
        text = explain.explain(facts)
        secs = time.time() - t0
        bad = unsupported(text, brief, today)
        n_bad = sum(len(v) for v in bad.values())
        bl = bottom_line_ok(text, facts)
        tm = type_mismatches(text, facts)
        ok = n_bad == 0 and bl is not False and not tm
        results.append({"appl_no": appl_no, "drug": facts["trade_name"], "status": facts["status"], "seconds": round(secs, 1),
                        "words": len(text.split()), "unsupported": bad, "n_unsupported": n_bad,
                        "bottom_line_ok": bl, "type_mismatches": tm, "pass": ok,
                        "disclaimer": "not legal advice" in text.lower(), "brief": brief, "text": text})
        print(f"{facts['trade_name'][:28]:28s} {facts['status'][:26]:26s} {secs:5.1f}s  "
              f"{'PASS' if ok else 'FAIL'}  unsupported={n_bad}{'' if not n_bad else bad}  "
              f"bottom_line={bl}  type_errors={tm or 0}")

    summary = {"model": explain.MODEL, "n": len(results),
               "passed_all_checks": sum(r["pass"] for r in results),
               "no_unsupported_items": sum(r["n_unsupported"] == 0 for r in results),
               "bottom_line_correct": sum(r["bottom_line_ok"] is not False for r in results),
               "no_type_errors": sum(not r["type_mismatches"] for r in results),
               "with_disclaimer": sum(r["disclaimer"] for r in results),
               "median_seconds": float(pd.Series([r["seconds"] for r in results]).median())}
    OUT.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"\n{summary}\nSaved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
