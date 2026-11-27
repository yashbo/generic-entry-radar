"""Load and clean the FDA Orange Book tilde-delimited data files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path(__file__).resolve().parent / "data" / "orange_book"
DATE_FMT = "%b %d, %Y"  # e.g. "Mar 21, 2025"


@dataclass
class OrangeBook:
    products: pd.DataFrame
    patents: pd.DataFrame
    exclusivity: pd.DataFrame


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="~", dtype=str, encoding="latin-1", keep_default_na=False)
    df.columns = [c.strip().lower().replace(";", "_").replace(" ", "_") for c in df.columns]
    for col in df.columns:
        df[col] = df[col].str.strip()
    df = df.replace({"": None})
    # Normalise keys so joins across the three files line up.
    if "appl_no" in df:
        df["appl_no"] = df["appl_no"].str.zfill(6)
    if "product_no" in df:
        df["product_no"] = df["product_no"].str.zfill(3)
    if "appl_type" in df:
        df["appl_type"] = df["appl_type"].str.upper()
    # Older editions write "SOLUTION; ORAL"; newer ones "SOLUTION;ORAL".
    if "df_route" in df:
        df["df_route"] = df["df_route"].str.replace(r"\s*;\s*", ";", regex=True).str.upper()
    return df


def _ensure(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    """Add columns that older editions lack (e.g. RS, Submission_Date)."""
    for col in cols:
        if col not in df:
            df[col] = None
    return df


def _date(s: pd.Series) -> pd.Series:
    # "Approved Prior to Jan 1, 1982" and other free text become NaT.
    return pd.to_datetime(s, format=DATE_FMT, errors="coerce")


def load(data_dir: Path = DEFAULT_DIR) -> OrangeBook:
    data_dir = Path(data_dir)
    missing = [f for f in ("products.txt", "patent.txt", "exclusivity.txt") if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {data_dir}. Run fetch_data.py first (see its docstring)."
        )

    products = _ensure(_read(data_dir / "products.txt"), ("rs", "te_code"))
    products["approval_date"] = _date(products["approval_date"])

    patents = _ensure(
        _read(data_dir / "patent.txt"),
        ("submission_date", "patent_use_code", "delist_flag", "drug_substance_flag", "drug_product_flag"),
    )
    patents["patent_expire_date"] = _date(patents["patent_expire_date_text"])
    patents["submission_date"] = _date(patents["submission_date"])
    # Pediatric exclusivity extensions appear as e.g. "7056942*PED" rows.
    patents["is_ped"] = patents["patent_no"].str.upper().str.endswith("*PED")
    patents["base_patent_no"] = patents["patent_no"].str.replace(r"\*PED$", "", regex=True, case=False)
    for flag in ("drug_substance_flag", "drug_product_flag", "delist_flag"):
        patents[flag] = patents[flag].fillna("").str.upper().eq("Y")

    exclusivity = _read(data_dir / "exclusivity.txt")
    exclusivity["exclusivity_date"] = _date(exclusivity["exclusivity_date"])

    return OrangeBook(products, patents, exclusivity)


def edition_date(ob: OrangeBook) -> pd.Timestamp | None:
    """Best estimate of when this edition was published: its most recent approval or patent submission.

    The data files carry no edition date of their own. Scoring as of this date (not the wall clock)
    keeps model numbers fixed until the data itself changes."""
    dates = [ob.products["approval_date"].max(), ob.patents["submission_date"].max()]
    dates = [d for d in dates if pd.notna(d)]
    return max(dates).normalize() if dates else None
