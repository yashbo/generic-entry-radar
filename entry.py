"""Estimate the earliest generic entry date for each NDA in the Orange Book.

This is a model of the *listed* protection only. Real entry dates also depend on
Paragraph IV litigation, settlements, 30-month stays, 180-day generic exclusivity,
patent term extensions not yet reflected, and unlisted patents (process patents).
Treat outputs as a screening signal, not legal or investment advice.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from orange_book import OrangeBook

ACTIVE_TYPES = {"RX", "OTC"}

# Exclusivities tied to a specific indication or label change. A generic can often
# launch with that use "carved out" of its label (section viii statement).
CARVEOUT_EXCL_PREFIXES = ("I-", "ODE", "D-", "NPP")

EXCL_DESCRIPTIONS = {
    "NCE": "New chemical entity (5 years)",
    "NP": "New product (3 years)",
    "PED": "Pediatric exclusivity (+6 months)",
    "NS": "New strength",
    "NDF": "New dosage form",
    "NR": "New route",
    "NC": "New combination",
    "GAIN": "Qualified infectious disease product (+5 years)",
}


def describe_exclusivity(code: str) -> str:
    if code in EXCL_DESCRIPTIONS:
        return EXCL_DESCRIPTIONS[code]
    if code.startswith("ODE"):
        return "Orphan drug exclusivity (7 years, indication-specific)"
    if code.startswith("I-"):
        return "New indication (3 years, indication-specific)"
    if code.startswith("D-"):
        return "New dosing schedule (3 years)"
    if code.startswith("NPP"):
        return "New patient population (3 years, label-specific)"
    if code.startswith("M-"):
        return "Miscellaneous exclusivity"
    return "Other exclusivity"


def _is_carveout_excl(code: pd.Series) -> pd.Series:
    return code.fillna("").str.upper().str.startswith(CARVEOUT_EXCL_PREFIXES)


def _join_unique(s: pd.Series) -> str:
    return "; ".join(sorted({x for x in s.dropna()}))


def _key(df: pd.DataFrame) -> pd.Series:
    return df["ingredient"].str.upper() + "|" + df["df_route"].str.upper()


def build_entry_table(ob: OrangeBook, today: pd.Timestamp | None = None) -> pd.DataFrame:
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    prod = ob.products.copy()
    prod["key"] = _key(prod)

    # --- NDA products, one row per application ---------------------------------
    nda = prod[prod["appl_type"] == "N"]
    nda_active = nda[nda["type"].isin(ACTIVE_TYPES)]
    apps = nda_active.groupby("appl_no").agg(
        trade_name=("trade_name", _join_unique),
        ingredient=("ingredient", _join_unique),
        df_route=("df_route", _join_unique),
        applicant=("applicant_full_name", _join_unique),
        n_strengths=("product_no", "nunique"),
        market=("type", _join_unique),
        nda_approval=("approval_date", "min"),
        keys=("key", lambda s: set(s)),
    )

    # --- Generic (ANDA) competition --------------------------------------------
    anda = prod[(prod["appl_type"] == "A") & prod["type"].isin(ACTIVE_TYPES)]
    gen_by_key = anda.groupby("key").agg(
        gen_first=("approval_date", "min"),
        gen_applicants=("applicant", lambda s: set(s.dropna())),
    )

    def generic_info(keys: set[str]) -> pd.Series:
        hit = gen_by_key.loc[gen_by_key.index.intersection(list(keys))]
        if hit.empty:
            return pd.Series({"generic_approved": False, "first_generic_approval": pd.NaT, "n_generic_makers": 0})
        makers = set().union(*hit["gen_applicants"])
        return pd.Series(
            {"generic_approved": True, "first_generic_approval": hit["gen_first"].min(), "n_generic_makers": len(makers)}
        )

    apps = apps.join(apps["keys"].apply(generic_info)).drop(columns="keys")

    # --- Patents ---------------------------------------------------------------
    pat = ob.patents[(ob.patents["appl_type"] == "N") & ~ob.patents["delist_flag"]].copy()
    pat["use_only"] = pat["patent_use_code"].notna() & ~pat["drug_substance_flag"] & ~pat["drug_product_flag"]
    d = pat["patent_expire_date"]
    pat["d_sub"] = d.where(pat["drug_substance_flag"])
    pat["d_prod"] = d.where(pat["drug_product_flag"])
    pat["d_use"] = d.where(pat["use_only"])
    pat["d_active"] = d.where(d >= today)
    pat["active_no"] = pat["base_patent_no"].where(d >= today)
    pats = pat.groupby("appl_no").agg(
        substance_patent_expiry=("d_sub", "max"),
        product_patent_expiry=("d_prod", "max"),
        use_patent_expiry=("d_use", "max"),
        latest_patent_expiry=("patent_expire_date", "max"),
        n_patents=("base_patent_no", "nunique"),
        n_active_patents=("active_no", "nunique"),
    )

    # --- Exclusivity -----------------------------------------------------------
    exc = ob.exclusivity[ob.exclusivity["appl_type"] == "N"].copy()
    exc["carveout"] = _is_carveout_excl(exc["exclusivity_code"])
    exc["d_hard"] = exc["exclusivity_date"].where(~exc["carveout"])
    exc["active_code"] = exc["exclusivity_code"].where(exc["exclusivity_date"] >= today)
    excs = exc.groupby("appl_no").agg(
        latest_exclusivity_expiry=("exclusivity_date", "max"),
        hard_exclusivity_expiry=("d_hard", "max"),
        active_exclusivities=("active_code", _join_unique),
    )

    t = apps.join(pats, how="left").join(excs, how="left")
    t["n_patents"] = t["n_patents"].fillna(0).astype(int)
    t["n_active_patents"] = t["n_active_patents"].fillna(0).astype(int)
    t["active_exclusivities"] = t["active_exclusivities"].fillna("")

    # Full listed protection (every listed patent and exclusivity must be cleared).
    t["entry_all_listed"] = t[["latest_patent_expiry", "latest_exclusivity_expiry"]].max(axis=1)
    # If method-of-use patents and indication-specific exclusivities are carved out.
    t["entry_with_carveout"] = t[
        ["substance_patent_expiry", "product_patent_expiry", "hard_exclusivity_expiry"]
    ].max(axis=1)

    t["main_barrier"] = t.apply(_main_barrier, axis=1)
    t["status"] = t.apply(lambda r: _status(r, today), axis=1)
    t["years_to_entry"] = ((t["entry_all_listed"] - today).dt.days / 365.25).clip(lower=0).round(1)
    t.loc[t["status"] == "Generic approved", "years_to_entry"] = 0.0
    return t.reset_index().sort_values("entry_all_listed")


def _main_barrier(r: pd.Series) -> str:
    candidates = {
        "Compound patent": r["substance_patent_expiry"],
        "Formulation patent": r["product_patent_expiry"],
        "Method-of-use patent": r["use_patent_expiry"],
        "Exclusivity": r["latest_exclusivity_expiry"],
    }
    candidates = {k: v for k, v in candidates.items() if pd.notna(v)}
    if not candidates:
        return "None listed"
    return max(candidates, key=candidates.get)


def _status(r: pd.Series, today: pd.Timestamp) -> str:
    if r["generic_approved"]:
        # FDA approval is not launch: settlements, carve-outs or at-risk launches
        # often explain an approved generic while patents are still listed.
        if pd.notna(r["entry_all_listed"]) and r["entry_all_listed"] >= today:
            return "Generic approved, patents still listed"
        return "Generic approved"
    if pd.isna(r["entry_all_listed"]) or r["entry_all_listed"] < today:
        return "Open: no listed barrier, no generic"
    if pd.isna(r["entry_with_carveout"]) or r["entry_with_carveout"] < today:
        return "Carve-out possible now"
    return "Protected"


def drug_facts(ob: OrangeBook, table: pd.DataFrame, appl_no: str) -> dict[str, Any]:
    """Everything known about one NDA, as plain data for the UI and the LLM."""
    row = table.loc[table["appl_no"] == appl_no].iloc[0]
    prods = ob.products[(ob.products["appl_type"] == "N") & (ob.products["appl_no"] == appl_no)]
    pats = ob.patents[(ob.patents["appl_type"] == "N") & (ob.patents["appl_no"] == appl_no)]
    excs = ob.exclusivity[(ob.exclusivity["appl_type"] == "N") & (ob.exclusivity["appl_no"] == appl_no)]

    patents = (
        pats.groupby(["patent_no"], as_index=False)
        .agg(
            expires=("patent_expire_date", "max"),
            substance=("drug_substance_flag", "any"),
            product=("drug_product_flag", "any"),
            use_codes=("patent_use_code", _join_unique),
            delisted=("delist_flag", "any"),
        )
        .sort_values("expires")
    )
    exclusivities = (
        excs.groupby("exclusivity_code", as_index=False)
        .agg(expires=("exclusivity_date", "max"))
        .assign(meaning=lambda d: d["exclusivity_code"].map(describe_exclusivity))
        .sort_values("expires")
    )

    def fmt(x: Any) -> Any:
        return x.strftime("%Y-%m-%d") if isinstance(x, pd.Timestamp) and pd.notna(x) else (None if pd.isna(x) else x)

    return {
        "appl_no": appl_no,
        "trade_name": row["trade_name"],
        "ingredient": row["ingredient"],
        "dosage_form_route": row["df_route"],
        "applicant": row["applicant"],
        "status": row["status"],
        "main_barrier": row["main_barrier"],
        "estimated_entry_all_listed": fmt(row["entry_all_listed"]),
        "estimated_entry_with_carveout": fmt(row["entry_with_carveout"]),
        "generic_approved": bool(row["generic_approved"]),
        "first_generic_approval": fmt(row["first_generic_approval"]),
        "n_generic_makers": int(row["n_generic_makers"]),
        "products": prods[["product_no", "strength", "type", "rld", "te_code"]]
        .astype(object).where(prods[["product_no", "strength", "type", "rld", "te_code"]].notna(), None)
        .to_dict("records"),
        "patents": [{k: fmt(v) for k, v in rec.items()} for rec in patents.to_dict("records")],
        "exclusivities": [{k: fmt(v) for k, v in rec.items()} for rec in exclusivities.to_dict("records")],
    }
