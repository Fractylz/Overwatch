"""Fit an asking-price model and evaluate on held-out reference numbers."""
from pathlib import Path
import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
CAT = ["brand", "model_family", "material", "condition", "box_papers"]
NUM = ["size_mm", "year"]
FIELDS = CAT + NUM


def train(csv, output):
    df = pd.read_csv(csv, dtype={"reference": str})
    required = set(FIELDS + ["price", "currency", "reference", "listing_id"])
    if required - set(df):
        raise ValueError(f"Missing columns: {sorted(required - set(df))}")
    if df.currency.isna().any() or len(df.currency.unique()) != 1:
        raise ValueError("Use one explicit currency throughout the training file.")
    df = df.drop_duplicates("listing_id").dropna(subset=["price", "brand", "model_family", "reference"]).copy()
    df = df.loc[np.isfinite(df.price) & (df.price > 0)].reset_index(drop=True)
    df[CAT] = df[CAT].fillna("Unknown")
    df[NUM] = df[NUM].apply(pd.to_numeric, errors="coerce")
    groups = df.brand + ":" + df.reference.str.upper()
    if groups.nunique() < 5:
        raise ValueError("At least five distinct watch references are required for a grouped holdout.")
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(df, groups=groups))
    prep = ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
                              ("numbers", "passthrough", NUM)])
    model = make_pipeline(prep, XGBRegressor(n_estimators=100, max_depth=3, learning_rate=.05,
                                              objective="reg:squarederror", n_jobs=2, random_state=42))
    X, y = df[FIELDS], df.price
    model.fit(X.iloc[tr], np.log1p(y.iloc[tr]))
    pred = np.expm1(model.predict(X.iloc[te]))
    report = dict(currency=str(df.currency.iloc[0]), rows=len(df), train_rows=len(tr), holdout_rows=len(te),
                  held_out_references=int(groups.iloc[te].nunique()),
                  mae=round(mean_absolute_error(y.iloc[te], pred), 2),
                  rmse=round(mean_squared_error(y.iloc[te], pred)**.5, 2),
                  median_baseline_mae=round(mean_absolute_error(y.iloc[te], np.repeat(y.iloc[tr].median(), len(te))), 2))
    model.fit(X, np.log1p(y))
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    joblib.dump(dict(model=model, report=report, catalogue=df.to_dict("records")), temp)
    os.replace(temp, output)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "data/watch_prices.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/price_model.joblib")
    args = parser.parse_args()
    train(args.csv, args.out)
