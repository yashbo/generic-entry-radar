"""FDA Paragraph IV certification list: patent challenges filed by generic makers.

A paragraph IV (PIV) certification means a generic applicant says a listed patent is
invalid or not infringed, which is the usual route to launching before patents expire.

Source: https://www.fda.gov/drugs/abbreviated-new-drug-application-anda/patent-certifications-and-suitability-petitions
("Paragraph IV List" Excel). fda.gov blocks scripted downloads: download it in a
browser and save it into data/piv/.

Point in time: only the DATE OF SUBMISSION (first PIV filing, recorded since
March 2004) is used as a model feature. Counts and 180-day outcomes describe the present
and would leak the future into the backtest, so they are display-only.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PIV_DIR = ROOT / "data" / "piv"

COLS = {
    "DRUG NAME": "drug_name",
    "DOSAGE FORM": "dosage_form",
    "STRENGTH": "strength",
    "RLD/NDA": "rld_nda",
    "DATE OF SUBMISSION": "piv_date",
    "NUMBER OF ANDAs SUBMITTED": "n_andas",
    "180-DAY STATUS": "status_180",
    "180-DAY DECISION POSTING DATE": "decision_180",
    "DATE OF FIRST APPLICANTAPPROVAL": "ftf_approval",
    "DATE OF FIRST COMMERCIAL MARKETING BY FTF": "ftf_marketing",
    "EXPIRATION DATE OF LAST QUALIFYING PATENT": "last_qualifying_patent",
}
DATE_COLS = ["piv_date", "decision_180", "ftf_approval", "ftf_marketing", "last_qualifying_patent"]
# "Pre-MMA": filed before the Medicare Modernization Act (enacted 8 Dec 2003). Exact date unknown.
PRE_MMA = pd.Timestamp("2003-12-08")


def _to_date(v) -> pd.Timestamp:
    if isinstance(v, pd.Timestamp) or hasattr(v, "year"):
        return pd.Timestamp(v)
    m = re.search(r"\d{1,2}/\d{1,2}/\d{4}", str(v or ""))
    return pd.to_datetime(m.group(0), format="%m/%d/%Y", errors="coerce") if m else pd.NaT


def latest_file(piv_dir: Path = PIV_DIR) -> Path | None:
    files = sorted(Path(piv_dir).glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_rows(path: Path) -> pd.DataFrame:
    """One row per list entry (drug + dosage form + strength), with parsed dates and NDAs."""
    raw = pd.read_excel(path, header=None, dtype=object)
    hdr = raw.index[raw[0].astype(str).str.strip().str.upper().eq("DRUG NAME")][0]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in raw.iloc[hdr]]
    df = df.rename(columns=COLS)[list(COLS.values())]
    df = df[df["drug_name"].notna()]
    df["piv_date_approx"] = df["piv_date"].astype(str).str.contains("Pre-MMA", case=False)
    df.loc[df["piv_date_approx"], "piv_date"] = PRE_MMA
    for c in DATE_COLS:
        df[c] = df[c].map(_to_date)
    df["n_andas"] = pd.to_numeric(df["n_andas"], errors="coerce")
    for c in ("drug_name", "dosage_form", "strength", "status_180"):
        df[c] = df[c].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    # "Ziagen\n20977" -> NDA 020977. A few entries list more than one RLD.
    df["appl_nos"] = df["rld_nda"].astype(str).map(lambda s: [n.zfill(6) for n in re.findall(r"\b\d{5,6}\b", s)])
    df["rld_name"] = df["rld_nda"].astype(str).str.split("\n").str[0].str.strip()
    return df.reset_index(drop=True)


def by_nda(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per NDA (earliest challenge, most ANDAs, etc.)."""
    x = rows.explode("appl_nos").rename(columns={"appl_nos": "appl_no"}).dropna(subset=["appl_no"])
    g = x.groupby("appl_no")
    out = pd.DataFrame({
        "piv_rld_name": g["rld_name"].first(),
        "piv_first_date": g["piv_date"].min(),
        "piv_latest_date": g["piv_date"].max(),
        "piv_entries": g.size(),
        "piv_max_andas": g["n_andas"].max(),
        "piv_status_180": g["status_180"].agg(lambda s: "; ".join(sorted(set(s.dropna())))),
        "piv_ftf_approval": g["ftf_approval"].min(),
        "piv_ftf_marketing": g["ftf_marketing"].min(),
        "piv_last_qualifying_patent": g["last_qualifying_patent"].max(),
    })
    return out.reset_index()


def load(piv_dir: Path = PIV_DIR) -> pd.DataFrame | None:
    """Per-NDA challenge table, or None when no list has been downloaded."""
    path = latest_file(piv_dir)
    return by_nda(load_rows(path)) if path else None


def piv_features(table: pd.DataFrame, piv: pd.DataFrame | None, as_of: pd.Timestamp) -> pd.DataFrame:
    """Point-in-time: was a PIV challenge on file before `as_of`, and for how long?"""
    out = pd.DataFrame({"appl_no": table["appl_no"].to_numpy()}, index=table.index)
    first = table["appl_no"].map(piv.set_index("appl_no")["piv_first_date"]) if piv is not None else pd.Series(pd.NaT, index=table.index)
    known = first.notna() & (first < pd.Timestamp(as_of))
    out["piv_filed"] = known.astype(int).to_numpy()
    out["years_since_piv"] = ((pd.Timestamp(as_of) - first).dt.days / 365.25).where(known, 0).round(2).to_numpy()
    return out
