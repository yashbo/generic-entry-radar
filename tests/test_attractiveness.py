"""Feature, label and scoring tests on the synthetic fixtures.
Run: python tests/test_attractiveness.py"""
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import attractiveness  # noqa: E402
import entry  # noqa: E402
import features  # noqa: E402
import orange_book  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
TODAY = pd.Timestamp("2026-11-26")
OB = orange_book.load(FIX / "orange_book")
TABLE = entry.build_entry_table(OB, today=TODAY)
F = features.build_features(OB, TABLE, TODAY).set_index("appl_no")


def test_feature_groups():
    assert F.loc["000001", "form_group"] == "tablet" and F.loc["000001", "route_group"] == "oral"
    assert F.loc["000004", "form_group"] == "injectable" and F.loc["000004", "route_group"] == "parenteral"
    assert F.loc["000005", "has_nce"] == 1 and F.loc["000001", "has_nce"] == 0


def test_generic_elsewhere_counts_makers_of_same_molecule():
    assert F.loc["000002", "n_generic_makers_ingredient"] == 2  # BETAZOLE: GENRX + CHEAPCO
    assert F.loc["000001", "ingredient_generic_elsewhere"] == 0


def test_pre_1982_approval_gets_an_age():
    # OLDDRUG is "Approved Prior to Jan 1, 1982": must not be NaN.
    assert F.loc["000002", "nda_age_years"] > 44


def test_features_are_point_in_time():
    # In the 2020 edition OMEGASTAT had no generic anywhere; by today it does.
    old = orange_book.load(FIX / "history" / "2020-01-01")
    as_of = pd.Timestamp("2020-01-01")
    f_old = features.build_features(old, entry.build_entry_table(old, today=as_of), as_of).set_index("appl_no")
    assert f_old.loc["000007", "n_generic_makers_ingredient"] == 0
    assert F.loc["000007", "n_generic_makers_ingredient"] == 1


# Synthetic Part D table: BLOCKED sells $5M in 2019 and $9M in 2023 (2023 must stay invisible to 2020).
SPEND = pd.DataFrame({
    "brand_key": ["BLOCKED", "BLOCKED", "OMEGASTAT", "PROTECTA"],
    "molecule_key": ["OMEGASTAT", "OMEGASTAT", "OMEGASTAT", "ALPHANIB"],
    "year": [2019, 2023, 2019, 2023],
    "spend": [5e6, 9e6, 1e6, 2e6],
})


def test_spend_is_point_in_time_and_matches_brand_and_molecule():
    import numpy as np
    import partd
    old = orange_book.load(FIX / "history" / "2020-01-01")
    as_of = pd.Timestamp("2020-01-01")
    t = entry.build_entry_table(old, today=as_of)
    f = partd.spend_features(t, SPEND, as_of).set_index("appl_no")
    assert partd.spend_year(SPEND, as_of) == 2019
    assert np.isclose(f.loc["000007", "log_brand_spend"], np.log10(1 + 5e6))
    assert np.isclose(f.loc["000007", "log_molecule_spend"], np.log10(1 + 6e6))  # brand + generic
    assert f.loc["000001", "log_brand_spend"] == 0  # PROTECTA only has 2023 sales
    # No spend year before 2011 data exists -> NaN, not zero.
    assert f.isna().sum().sum() == 0
    assert partd.spend_features(t, SPEND, pd.Timestamp("2015-01-01"))["log_brand_spend"].isna().all()


def test_molecule_key_ignores_order_and_separators():
    import partd
    assert partd.molecule_key("Abacavir/Lamivudine") == partd.molecule_key("LAMIVUDINE; ABACAVIR")


def test_labels_and_scoring_end_to_end():
    data = attractiveness.build_dataset(FIX / "history", FIX / "orange_book", observed_until=TODAY, spend_long=SPEND, piv_table=None)
    labels = data.set_index("appl_no")["y"].to_dict()
    # 000007: generic 2025-01, 7 months after 2024-06 expiry -> 1. 000003: never -> 0.
    assert labels == {"000007": 1, "000003": 0}
    assert data["log_brand_spend"].notna().all()
    bundle = attractiveness.fit_final(data)
    p = attractiveness.score(bundle, OB, TABLE, TODAY, spend_long=SPEND, piv_table=None)
    cohort = set(TABLE.loc[TABLE["status"].isin(["Protected", "Carve-out possible now"]), "appl_no"])
    assert set(p.index) == cohort
    assert p.between(0, 1).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("pass", name)
