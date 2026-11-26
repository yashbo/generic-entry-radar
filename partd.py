"""Medicare Part D spending by drug, as point-in-time features for Orange Book NDAs.

Sources (see fetch_partd.py): four CMS releases covering calendar years 2011-2024.
Where releases overlap, the newer release wins (CMS revises figures).

Matching to the Orange Book:
  * brand spend    : Part D brand name == an NDA trade name (normalised)
  * molecule spend : Part D generic name == NDA ingredient (same set of words), summed
                     over every brand and generic product of that molecule (market size)
Point in time: an edition dated E only sees the latest spending year <= E.year - 1.
Part D covers retail prescriptions, so clinic-administered (Part B) and OTC drugs
show little or no spend; that absence is itself informative.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PARTD_DIR = ROOT / "data" / "partd"
CACHE = PARTD_DIR / "spend_long.csv"


def norm_name(s: str | None) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", str(s or "").upper()).strip()


def molecule_key(s: str | None) -> str:
    """Order-independent key: 'Abacavir/Lamivudine' == 'LAMIVUDINE; ABACAVIR'."""
    return " ".join(sorted(set(norm_name(s).split())))


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def _xlsx(zip_path: Path, sheet: str) -> pd.DataFrame:
    zf = zipfile.ZipFile(zip_path)
    return pd.read_excel(io.BytesIO(zf.read(zf.namelist()[0])), sheet_name=sheet, header=None)


def _from_2015(path: Path) -> pd.DataFrame:
    raw = _xlsx(path, "Data")
    hdr_row = raw.index[raw[0].astype(str).str.strip().eq("Brand Name")][0]
    df = raw.iloc[hdr_row + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[hdr_row]]
    out = []
    for col in df.columns:
        m = re.fullmatch(r"Total Spending, (\d{4})", col)
        if m:
            out.append(pd.DataFrame({"brand": df["Brand Name"], "generic": df["Generic Name"],
                                     "year": int(m.group(1)), "spend": _num(df[col])}))
    return pd.concat(out)


def _from_2016(path: Path) -> pd.DataFrame:
    raw = _xlsx(path, "Spending Utilization YTD 2016")
    hdr_row = raw.index[raw[0].astype(str).str.strip().eq("Brand Name")][0]
    years = [int(y) for y in re.findall(r"Calendar Year (\d{4})", " ".join(raw.iloc[hdr_row - 1].dropna().astype(str)))]
    header = raw.iloc[hdr_row].astype(str).str.strip().tolist()
    spend_cols = [i for i, h in enumerate(header) if h == "Total Spending"][: len(years)]
    df = raw.iloc[hdr_row + 1:]
    return pd.concat(
        pd.DataFrame({"brand": df[0], "generic": df[1], "year": y, "spend": _num(df[i])})
        for y, i in zip(years, spend_cols)
    )


def _from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = df[df["Mftr_Name"].str.strip().str.upper().eq("OVERALL")]
    out = []
    for col in df.columns:
        m = re.fullmatch(r"Tot_Spndng_(\d{4})", col)
        if m:
            out.append(pd.DataFrame({"brand": df["Brnd_Name"], "generic": df["Gnrc_Name"],
                                     "year": int(m.group(1)), "spend": _num(df[col])}))
    return pd.concat(out)


# Oldest release first; later ones override overlapping years.
READERS = [
    ("Medicare_PartD_All_Drugs_2015.zip", _from_2015),
    ("Medicare_Data_PartD_2016.zip", _from_2016),
    ("DSD_PTD_R22_P04_V22_D21_BGM.csv", _from_csv),
    ("DSD_PTD_RY26_P04_V10_DY24_BGM.csv", _from_csv),
]


def build_long(partd_dir: Path = PARTD_DIR) -> pd.DataFrame:
    parts = []
    for rank, (name, reader) in enumerate(READERS):
        p = Path(partd_dir) / name
        if p.exists():
            parts.append(reader(p).assign(release=rank))
    if not parts:
        raise FileNotFoundError(f"No Part D files in {partd_dir}. Run fetch_partd.py first.")
    df = pd.concat(parts, ignore_index=True)
    df["brand_key"] = df["brand"].map(norm_name)
    df["molecule_key"] = df["generic"].map(molecule_key)
    df = df[(df["brand_key"] != "") & df["spend"].notna()]
    # Newest release wins for each (brand, molecule, year).
    df = df.sort_values("release").drop_duplicates(["brand_key", "molecule_key", "year"], keep="last")
    return df[["brand_key", "molecule_key", "year", "spend"]].reset_index(drop=True)


def load_long(partd_dir: Path = PARTD_DIR, rebuild: bool = False) -> pd.DataFrame | None:
    """Cached long table, or None when no Part D data has been downloaded."""
    cache = Path(partd_dir) / CACHE.name
    if cache.exists() and not rebuild:
        return pd.read_csv(cache, dtype={"brand_key": str, "molecule_key": str})
    try:
        df = build_long(partd_dir)
    except FileNotFoundError:
        return None
    df.to_csv(cache, index=False)
    return df


def spend_year(long: pd.DataFrame, as_of: pd.Timestamp) -> int | None:
    years = long.loc[long["year"] <= pd.Timestamp(as_of).year - 1, "year"]
    return int(years.max()) if len(years) else None


def spend_features(table: pd.DataFrame, long: pd.DataFrame | None, as_of: pd.Timestamp) -> pd.DataFrame:
    """Per-NDA spend features for the given edition date (NaN when no data year exists)."""
    out = pd.DataFrame({"appl_no": table["appl_no"].to_numpy()}, index=table.index)
    year = spend_year(long, as_of) if long is not None else None
    if year is None:
        out["log_brand_spend"] = np.nan
        out["log_molecule_spend"] = np.nan
        return out
    y = long[long["year"] == year]
    brand = y.groupby("brand_key")["spend"].sum()
    molecule = y.groupby("molecule_key")["spend"].sum()

    def brand_total(names: str) -> float:
        keys = {norm_name(n) for n in str(names or "").split("; ")}
        return float(sum(brand.get(k, 0.0) for k in keys))

    def molecule_total(ingredients: str) -> float:
        keys = {molecule_key(i) for i in str(ingredients or "").split("; ")}
        return float(sum(molecule.get(k, 0.0) for k in keys))

    out["log_brand_spend"] = np.log10(1 + table["trade_name"].map(brand_total).to_numpy())
    out["log_molecule_spend"] = np.log10(1 + table["ingredient"].map(molecule_total).to_numpy())
    return out
