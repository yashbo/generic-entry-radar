"""The hosted app's numbers must not drift with the wall clock while the data is unchanged.
Run: python tests/test_radar.py"""
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import attractiveness  # noqa: E402
import orange_book  # noqa: E402
import piv  # noqa: E402
import radar  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "orange_book"
OB = orange_book.load(FIX)


def test_edition_date_is_latest_record():
    # Fixture: latest approval or patent submission in the three files.
    latest = max(OB.products["approval_date"].max(), OB.patents["submission_date"].max())
    assert orange_book.edition_date(OB) == latest.normalize()


def test_likelihood_does_not_drift_with_today():
    if not attractiveness.MODEL_PATH.exists():
        return
    edition = orange_book.edition_date(OB)
    _, a = radar.build_table(FIX, edition + pd.DateOffset(days=30))
    _, b = radar.build_table(FIX, edition + pd.DateOffset(days=200))
    assert a.attrs["model_as_of"] == b.attrs["model_as_of"] == edition.strftime("%Y-%m-%d")
    both = a[["appl_no", "p_generic"]].merge(b[["appl_no", "p_generic"]], on="appl_no").dropna()
    assert len(both) > 0
    assert (both["p_generic_x"] == both["p_generic_y"]).all()


def test_piv_list_date_from_file_name():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "10-27-2026 Paragraph IV List (Public).xlsx").write_bytes(b"")
        assert piv.list_date(Path(d)) == pd.Timestamp("2026-10-27")
        assert piv.list_date(Path(d) / "missing") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("pass", name)
