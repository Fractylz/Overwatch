"""
Phase 2 - Price estimator.

Trains XGBoost (plus Linear Regression and Random Forest baselines, so you have a
comparison table for your evaluation chapter) to estimate a watch's CURRENT market
value from tabular features. Expects a CSV like:

    brand,model_family,year,condition,material,size_mm,box_papers,price
    Rolex,Submariner,2019,Good,Steel,40,Yes,32000
    Omega,Speedmaster,2015,Fair,Steel,42,No,3800
    Cartier,Santos,2021,Excellent,Gold,40,Yes,21000
    ...

'price' is the target (what you scraped the watch sold/listed for).
Aim for a few hundred+ rows. Outputs: artifacts/price_model.joblib

Install:
    pip install pandas scikit-learn xgboost joblib
"""

from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from xgboost import XGBRegressor

# ---------------- config ----------------
CSV = "data/watch_prices.csv"
TARGET = "price"
CAT = ["brand", "model_family", "condition", "material", "box_papers"]
NUM = ["year", "size_mm"]
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

# ---------------- data ----------------
df = pd.read_csv(CSV)
X, y = df[CAT + NUM], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# one-hot the categoricals, scale the numerics; handle_unknown keeps unseen brands from crashing
pre = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), NUM),
    ]
)

candidates = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    ),
}


def report(name, pipe):
    pred = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    print(
        f"{name:16s}  MAE={mean_absolute_error(y_test, pred):10.1f}  "
        f"RMSE={rmse:10.1f}  R2={r2_score(y_test, pred):6.3f}  "
        f"MAPE={mean_absolute_percentage_error(y_test, pred) * 100:5.1f}%"
    )


print(f"{len(df)} rows | target='{TARGET}'\n")
best = None
for name, reg in candidates.items():
    pipe = Pipeline([("pre", pre), ("reg", reg)]).fit(X_train, y_train)
    report(name, pipe)
    if name == "XGBoost":
        best = pipe

joblib.dump(best, OUT / "price_model.joblib")
print(f"\nSaved best model (XGBoost) -> {OUT/'price_model.joblib'}")
