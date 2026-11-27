"""Generic-attractiveness model: will a generic actually arrive once protection ends?

Label (from the backtest): for a drug with no generic at edition date E, did a generic
get approved no later than GENERIC_WINDOW_YEARS after its predicted all-listed entry
date? Only matured predictions are labeled, so every label is fully observed.

Evaluation:
  * grouped 5-fold CV (a drug never sits in both train and test folds)
  * temporal holdout: train on editions before 2016, test on 2016+ editions,
    restricted to drugs never seen in training
Baselines: base rate, and a one-feature model ("molecule already has generics elsewhere").

Run: python attractiveness.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import backtest
import entry
import features
import orange_book
import partd
import piv

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "data" / "model"
MODEL_PATH = MODEL_DIR / "attractiveness.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
GENERIC_WINDOW_YEARS = 2
TEMPORAL_SPLIT = pd.Timestamp("2016-01-01")
SEED = 0
# Chosen over gradient boosting: better temporal-holdout AUC and calibration (see metrics.json).
FINAL_MODEL = "logistic_regression"


# ----------------------------------------------------------------- dataset
def label_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep matured, clean backtest rows and add the binary label `y`."""
    pred = rows["pred_all_listed"]
    matured = pred <= rows["observed_until"] - pd.DateOffset(years=backtest.MATURITY_YEARS)
    r = rows[matured & ~rows["prior_anda"]].copy()
    deadline = r["pred_all_listed"] + pd.DateOffset(years=GENERIC_WINDOW_YEARS)
    r["y"] = (r["actual_first_generic"].notna() & (r["actual_first_generic"] <= deadline)).astype(int)
    return r


_AUTO = object()


def build_dataset(hist_dir: Path = backtest.HIST_DIR, current_dir: Path = orange_book.DEFAULT_DIR,
                  observed_until: pd.Timestamp | None = None, spend_long=_AUTO, piv_table=_AUTO) -> pd.DataFrame:
    """spend_long: Part D long table; default loads data/partd, None means no spend data."""
    current = orange_book.load(current_dir)
    if spend_long is _AUTO:
        spend_long = partd.load_long()
    if piv_table is _AUTO:
        piv_table = piv.load()
    if observed_until is None:
        observed_until = current.products.loc[current.products["appl_type"] == "A", "approval_date"].max()
    parts = []
    for d in sorted(p for p in Path(hist_dir).iterdir() if p.is_dir()):
        as_of = pd.Timestamp(d.name)
        ob = orange_book.load(d)
        table = entry.build_entry_table(ob, today=as_of)
        feats = features.build_features(ob, table, as_of, spend_long, piv_table)
        rows = label_rows(backtest.backtest_edition(ob, as_of, current, observed_until))
        parts.append(rows[["as_of", "appl_no", "trade_name", "ingredient", "y"]].merge(feats, on="appl_no"))
    data = pd.concat(parts, ignore_index=True)
    for c in features.CATEGORICAL:  # concat can drop category dtype when categories differ
        data[c] = data[c].astype("category")
    return data


# ----------------------------------------------------------------- models
def make_models() -> dict:
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=1.0, categorical_features="from_dtype", random_state=SEED,
    )
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), features.CATEGORICAL),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), features.NUMERIC),
    ])
    logit = make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))
    form_only = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["form_group", "route_group"])]),
        LogisticRegression(max_iter=2000),
    )
    one = make_pipeline(StandardScaler(), LogisticRegression())
    spend_only = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression())
    return {
        FINAL_MODEL: (logit, features.FEATURES),
        "gradient_boosting": (hgb, features.FEATURES),
        "baseline_form_and_route": (form_only, ["form_group", "route_group"]),
        "baseline_spend_only": (spend_only, features.SPEND),
        "baseline_generic_elsewhere": (one, ["ingredient_generic_elsewhere"]),
    }


def _scores(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 3),
        "avg_precision": round(float(average_precision_score(y, p)), 3),
        "brier": round(float(brier_score_loss(y, p)), 3),
    }


def evaluate(data: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    y = data["y"].to_numpy()
    results: dict = {"n_rows": len(data), "n_drugs": int(data["appl_no"].nunique()),
                     "positive_rate": round(float(y.mean()), 3), "cv": {}, "temporal": {}}
    results["cv"]["base_rate"] = {**_scores(y, np.full(len(y), y.mean())), "roc_auc": 0.5}
    oof = {}

    # Grouped CV: all editions of one drug stay in the same fold.
    gkf = GroupKFold(n_splits=5)
    for name, (model, cols) in make_models().items():
        p = np.zeros(len(data))
        for tr, te in gkf.split(data, y, groups=data["appl_no"]):
            model.fit(data.iloc[tr][cols], y[tr])
            p[te] = model.predict_proba(data.iloc[te][cols])[:, 1]
        oof[name] = p
        results["cv"][name] = _scores(y, p)

    # Temporal holdout, unseen drugs only.
    train = data[data["as_of"] < TEMPORAL_SPLIT]
    test = data[(data["as_of"] >= TEMPORAL_SPLIT) & ~data["appl_no"].isin(train["appl_no"])]
    results["temporal"]["n_train"], results["temporal"]["n_test"] = len(train), len(test)
    importance = pd.DataFrame()
    if len(test) >= 30 and test["y"].nunique() == 2:
        for name, (model, cols) in make_models().items():
            model.fit(train[cols], train["y"])
            results["temporal"][name] = _scores(test["y"].to_numpy(), model.predict_proba(test[cols])[:, 1])
            if name == FINAL_MODEL:
                pi = permutation_importance(model, test[cols], test["y"], scoring="roc_auc",
                                            n_repeats=10, random_state=SEED)
                importance = (pd.DataFrame({"feature": cols, "auc_drop": pi.importances_mean, "std": pi.importances_std})
                              .sort_values("auc_drop", ascending=False).round(4))

    # Calibration of the final model's out-of-fold probabilities.
    calib = (pd.DataFrame({"p": oof[FINAL_MODEL], "y": y})
             .assign(bin=lambda d: pd.qcut(d["p"], 10, duplicates="drop"))
             .groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
             .round(3).reset_index(drop=True))
    return results, importance, calib


# ----------------------------------------------------------------- fit / save / score
def fit_final(data: pd.DataFrame) -> dict:
    model, cols = make_models()[FINAL_MODEL]
    model.fit(data[cols], data["y"])
    return {
        "model": model,
        "features": cols,
        "categories": {c: list(data[c].cat.categories) for c in features.CATEGORICAL},
        "trained_rows": len(data),
        "window_years": GENERIC_WINDOW_YEARS,
        "model_name": FINAL_MODEL,
    }


def load_model(path: Path = MODEL_PATH) -> dict | None:
    return joblib.load(path) if Path(path).exists() else None


def score(bundle: dict, ob: orange_book.OrangeBook, table: pd.DataFrame, as_of: pd.Timestamp,
          spend_long=_AUTO, piv_table=_AUTO) -> pd.Series:
    """P(generic within the window after protection ends), indexed by appl_no.

    Only drugs without a generic (Protected / Carve-out possible now) are scored: that
    is the population the model was trained on.
    """
    cohort = table[table["status"].isin(backtest.COHORT)]
    if spend_long is _AUTO:
        spend_long = partd.load_long()
    if piv_table is _AUTO:
        piv_table = piv.load()
    f = features.build_features(ob, cohort, as_of, spend_long, piv_table)
    for c, cats in bundle["categories"].items():
        f[c] = pd.Categorical(f[c].astype(str), categories=cats)
    p = bundle["model"].predict_proba(f[bundle["features"]])[:, 1]
    return pd.Series(p, index=f["appl_no"].to_numpy(), name="p_generic")


def main() -> None:
    print("Building point-in-time dataset from archived editions ...")
    data = build_dataset()
    results, importance, calib = evaluate(data)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fit_final(data), MODEL_PATH)
    results["importance"] = importance.to_dict("records")
    results["calibration"] = calib.to_dict("records")
    METRICS_PATH.write_text(json.dumps(results, indent=2, default=str))
    data.to_csv(MODEL_DIR / "training_data.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\nRows {results['n_rows']}  drugs {results['n_drugs']}  positive rate {results['positive_rate']}")
    print("\nGrouped 5-fold CV:\n" + pd.DataFrame(results["cv"]).T.to_string())
    t = {k: v for k, v in results["temporal"].items() if isinstance(v, dict)}
    print(f"\nTemporal holdout (train <2016: {results['temporal']['n_train']} rows, "
          f"test 2016+ unseen drugs: {results['temporal']['n_test']} rows):\n" + pd.DataFrame(t).T.to_string())
    print("\nPermutation importance (temporal test, AUC drop):\n" + importance.head(10).to_string(index=False))
    print("\nCalibration (CV deciles):\n" + calib.to_string(index=False))
    print(f"\nSaved {MODEL_PATH}")


if __name__ == "__main__":
    main()
