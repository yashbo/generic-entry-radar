"""Point-in-time features describing how attractive an NDA is to generic makers.

Every feature is computed from ONE Orange Book edition as of its date, so features for
an archived edition never see information from later editions (no look-ahead leakage).
"""
from __future__ import annotations

import pandas as pd

import entry
import partd
import piv
from orange_book import OrangeBook

PRE_1982 = pd.Timestamp("1982-01-01")  # "Approved Prior to Jan 1, 1982" parses to NaT

FORM_GROUPS = [
    ("TABLET", "tablet"), ("CAPSULE", "capsule"), ("INJECT", "injectable"), ("EMULSION", "injectable"),
    ("AEROSOL", "inhaled/spray"), ("SPRAY", "inhaled/spray"), ("INHAL", "inhaled/spray"),
    ("CREAM", "topical semisolid"), ("OINTMENT", "topical semisolid"), ("GEL", "topical semisolid"),
    ("LOTION", "topical semisolid"), ("FOAM", "topical semisolid"),
    ("PATCH", "patch/film/implant"), ("FILM", "patch/film/implant"), ("IMPLANT", "patch/film/implant"),
    ("INSERT", "patch/film/implant"), ("RING", "patch/film/implant"), ("SYSTEM", "patch/film/implant"),
    ("SUSPENSION", "liquid"), ("SOLUTION", "liquid"), ("SYRUP", "liquid"), ("ELIXIR", "liquid"),
    ("POWDER", "powder"), ("GRANULE", "powder"),
]
ROUTE_GROUPS = [
    ("ORAL", "oral"), ("SUBLINGUAL", "oral"), ("BUCCAL", "oral"),
    ("INTRAVENOUS", "parenteral"), ("INJECTION", "parenteral"), ("SUBCUTANEOUS", "parenteral"),
    ("INTRAMUSCULAR", "parenteral"), ("INTRATHECAL", "parenteral"),
    ("TOPICAL", "topical"), ("TRANSDERMAL", "transdermal"), ("OPHTHALMIC", "ophthalmic"),
    ("INHALATION", "inhalation"), ("NASAL", "nasal"), ("VAGINAL", "vaginal/rectal"), ("RECTAL", "vaginal/rectal"),
]

CATEGORICAL = ["form_group", "route_group", "main_barrier"]
NUMERIC = [
    "modified_release", "is_combo", "is_otc", "n_strengths", "nda_age_years", "years_to_entry",
    "n_patents", "n_active_patents", "has_orphan", "has_nce", "ingredient_generic_elsewhere",
    "n_generic_makers_ingredient", "n_brand_ndas_ingredient", "applicant_portfolio",
]
# Medicare Part D spend in the last full year before the edition (NaN when unavailable).
SPEND = ["log_brand_spend", "log_molecule_spend"]
# Paragraph IV challenge on file before the edition date.
PIV = ["piv_filed", "years_since_piv"]
NUMERIC = NUMERIC + SPEND + PIV
FEATURES = CATEGORICAL + NUMERIC


def _group(text: str, groups: list[tuple[str, str]]) -> str:
    for needle, label in groups:
        if needle in text:
            return label
    return "other"


def _first(s: str | None) -> str:
    return (s or "").split("; ")[0].upper()


def build_features(ob: OrangeBook, table: pd.DataFrame, as_of: pd.Timestamp,
                   spend_long: pd.DataFrame | None = None, piv_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per NDA in `table` (from entry.build_entry_table for the same edition)."""
    as_of = pd.Timestamp(as_of)
    prod = ob.products
    active = prod[prod["type"].isin(entry.ACTIVE_TYPES)]

    anda = active[active["appl_type"] == "A"]
    makers_by_ingr = anda.groupby(anda["ingredient"].str.upper())["applicant"].nunique()
    nda = active[active["appl_type"] == "N"]
    brands_by_ingr = nda.groupby(nda["ingredient"].str.upper())["appl_no"].nunique()
    portfolio = nda.groupby("applicant_full_name")["appl_no"].nunique()

    exc = ob.exclusivity[ob.exclusivity["appl_type"] == "N"]
    code = exc["exclusivity_code"].fillna("").str.upper()
    orphan_apps = set(exc.loc[code.str.startswith("ODE"), "appl_no"])
    nce_apps = set(exc.loc[code.eq("NCE"), "appl_no"])

    t = table
    df_route = t["df_route"].map(_first)
    form = df_route.str.split(";").str[0]
    route = df_route.str.split(";").str[-1]
    ingredient = t["ingredient"].map(_first)
    applicant = t["applicant"].fillna("").str.split("; ").str[0]

    f = pd.DataFrame(index=t.index)
    f["appl_no"] = t["appl_no"]
    f["form_group"] = form.map(lambda s: _group(s, FORM_GROUPS))
    f["route_group"] = route.map(lambda s: _group(s, ROUTE_GROUPS))
    f["main_barrier"] = t["main_barrier"]
    f["modified_release"] = form.str.contains("EXTENDED|DELAYED|ORALLY DISINTEGRATING", regex=True).astype(int)
    f["is_combo"] = ingredient.str.contains(";").astype(int)
    f["is_otc"] = t["market"].fillna("").str.contains("OTC").astype(int)
    f["n_strengths"] = t["n_strengths"]
    f["nda_age_years"] = ((as_of - t["nda_approval"].fillna(PRE_1982)).dt.days / 365.25).round(2)
    f["years_to_entry"] = ((t["entry_all_listed"] - as_of).dt.days / 365.25).clip(lower=0).fillna(0).round(2)
    f["n_patents"] = t["n_patents"]
    f["n_active_patents"] = t["n_active_patents"]
    f["has_orphan"] = t["appl_no"].isin(orphan_apps).astype(int)
    f["has_nce"] = t["appl_no"].isin(nce_apps).astype(int)
    # Generics of the same molecule in other forms: proves manufacturability and interest.
    f["n_generic_makers_ingredient"] = ingredient.map(makers_by_ingr).fillna(0).astype(int)
    f["ingredient_generic_elsewhere"] = (f["n_generic_makers_ingredient"] > 0).astype(int)
    f["n_brand_ndas_ingredient"] = ingredient.map(brands_by_ingr).fillna(1).astype(int)
    f["applicant_portfolio"] = applicant.map(portfolio).fillna(1).astype(int)
    sp = partd.spend_features(table, spend_long, as_of)
    for c in SPEND:
        f[c] = sp[c]
    pf = piv.piv_features(table, piv_table, as_of)
    for c in PIV:
        f[c] = pf[c]
    for c in CATEGORICAL:
        f[c] = f[c].astype("category")
    return f
