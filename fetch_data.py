"""Get the FDA Orange Book data files into data/orange_book/.

Source page: https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files
The zip contains tilde-delimited products.txt, patent.txt and exclusivity.txt.

FDA's site often blocks scripted downloads, so the usual path is:
  1. Download the "Orange Book Data Files (compressed)" zip in your browser.
  2. Put it in generic_entry_radar/data/ (any name ending in .zip), or pass --zip PATH.
  3. Run: python generic_entry_radar/fetch_data.py
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests

ORANGE_BOOK_URL = "https://www.fda.gov/media/76860/download"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "orange_book"
EXPECTED = {"products.txt", "patent.txt", "exclusivity.txt"}


def extract(zip_bytes: bytes, dest: Path = DATA_DIR) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = {Path(n).name.lower(): n for n in zf.namelist()}
        missing = EXPECTED - names.keys()
        if missing:
            raise RuntimeError(f"Zip is missing {missing}; contents: {list(names)}")
        for short in EXPECTED:
            # Write by basename only, so no path inside the archive can escape dest.
            (dest / short).write_bytes(zf.read(names[short]))
    print(f"Extracted {sorted(EXPECTED)} to {dest}")
    return dest


def try_download() -> bytes:
    print(f"Trying {ORANGE_BOOK_URL} ...")
    resp = requests.get(
        ORANGE_BOOK_URL,
        headers={"User-Agent": "generic-entry-radar/0.1 (research script)"},
        timeout=120,
    )
    resp.raise_for_status()
    if not resp.content.startswith(b"PK"):
        raise RuntimeError("FDA returned a web page instead of the zip (scripted downloads blocked).")
    return resp.content


def find_local_zip() -> Path | None:
    zips = sorted((ROOT / "data").glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, help="Path to a manually downloaded Orange Book zip")
    args = ap.parse_args()

    local = args.zip or find_local_zip()
    if local:
        print(f"Using local zip {local}")
        extract(local.read_bytes())
        return 0
    try:
        extract(try_download())
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "\nDownload the zip in your browser from\n"
            "  https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files\n"
            f"and save it into {ROOT / 'data'}, then rerun this script.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
