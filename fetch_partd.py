"""Download Medicare Part D spending-by-drug releases (archived CMS files).

data.cms.gov refuses some networks, so these come from the Internet Archive's copies
of CMS's own published files. Together they cover spending years 2011-2024.
Files are saved unmodified to data/partd/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
PARTD_DIR = ROOT / "data" / "partd"
CMS_OLD = "www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Information-on-Prescription-Drugs/Downloads/"

FILES = {  # local name: (wayback timestamp, original URL)
    "Medicare_PartD_All_Drugs_2015.zip": ("20190322074501", CMS_OLD + "Medicare_PartD_All_Drugs_2015.zip"),
    "Medicare_Data_PartD_2016.zip": ("20190322074451", CMS_OLD + "Medicare_Data_PartD_2016.zip"),
    "DSD_PTD_R22_P04_V22_D21_BGM.csv": (
        "20230304061912",
        "data.cms.gov/sites/default/files/2023-02/128e36b3-e62d-49cf-9963-dc6b1e540cc9/DSD_PTD_R22_P04_V22_D21_BGM.csv",
    ),
    "DSD_PTD_RY26_P04_V10_DY24_BGM.csv": (
        "20260806142415",
        "data.cms.gov/sites/default/files/2026-06/98218f98-166c-4723-8438-c344a4ef96a6/DSD_PTD_RY26_P04_V10_DY24_BGM.csv",
    ),
}


def main() -> int:
    PARTD_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, (ts, url) in FILES.items():
        dest = PARTD_DIR / name
        if dest.exists():
            print(f"{name}: already present")
            continue
        for attempt in range(3):
            try:
                r = requests.get(f"https://web.archive.org/web/{ts}id_/https://{url}", timeout=300)
                r.raise_for_status()
                ok = r.content.startswith(b"PK") if name.endswith(".zip") else b"," in r.content[:2000]
                if not ok:
                    raise RuntimeError("unexpected content")
                dest.write_bytes(r.content)
                print(f"{name}: {len(r.content) / 1e6:.2f} MB")
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    failures += 1
                    print(f"{name}: FAILED {exc}", file=sys.stderr)
                time.sleep(3)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
