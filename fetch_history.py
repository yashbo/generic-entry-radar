"""Download archived Orange Book editions from the Internet Archive for backtesting.

Each snapshot is extracted to data/history/<YYYY-MM-DD>/ (the snapshot date, which is
used as the "as of" date of that edition).
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
HIST_DIR = ROOT / "data" / "history"
OLD_URL = "www.fda.gov/downloads/Drugs/InformationOnDrugs/UCM163762.zip"
NEW_URL = "fda.gov/media/76860/download"

# Wayback timestamps found via the CDX API (one capture per year).
SNAPSHOTS = {
    "20100113002223": OLD_URL,
    "20120126210607": OLD_URL,
    "20140915104404": OLD_URL,
    "20160407115331": OLD_URL,
    "20180125053257": OLD_URL,
    "20201112023717": NEW_URL,
    "20220309013510": NEW_URL,
}
WANTED = {"products.txt", "patent.txt", "exclusivity.txt"}


def fetch_one(ts: str, url: str) -> Path:
    dest = HIST_DIR / f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
    if all((dest / f).exists() for f in WANTED):
        print(f"{dest.name}: already present")
        return dest
    # "id_" asks the Wayback Machine for the original bytes, without its banner.
    r = requests.get(f"https://web.archive.org/web/{ts}id_/https://{url}", timeout=180)
    r.raise_for_status()
    if not r.content.startswith(b"PK"):
        raise RuntimeError(f"{ts}: response is not a zip")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = {Path(n).name.lower(): n for n in zf.namelist()}
        missing = WANTED - names.keys()
        if missing:
            raise RuntimeError(f"{ts}: zip lacks {missing}; has {list(names)}")
        for short in WANTED:
            (dest / short).write_bytes(zf.read(names[short]))
    print(f"{dest.name}: {len(r.content) / 1e6:.2f} MB extracted")
    return dest


def main() -> int:
    failures = 0
    for ts, url in SNAPSHOTS.items():
        try:
            fetch_one(ts, url)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"{ts}: FAILED {exc}", file=sys.stderr)
        time.sleep(1)  # be gentle with archive.org
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
