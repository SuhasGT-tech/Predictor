"""
Train a price-prediction model per market (Arasikere, Tiptur) using the
feature table from step 3.

MODEL CHOICE
------------
Tries XGBoost first (generally stronger for tabular data with this many
features), falls back to RandomForest if xgboost isn't installed - so
this runs either way.

WHY TIME-SERIES SPLIT, NOT RANDOM SPLIT
Shuffling rows randomly before splitting train/test would leak future
information into training (e.g. training on a Tuesday price to predict
the previous Monday). We always train on the past and test on the most
recent slice, which mirrors how you'll actually use the model day to day.

OUTPUT
------
  models/model_<market>.joblib       - trained model
  models/feature_cols_<market>.json  - exact feature list/order the model expects
  models/metrics_<market>.json       - validation performance (MAE, MAPE)

USAGE
-----
    python 4_train_model.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import joblib

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(BASE_DIR, "data", "features.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TARGET = "Modal"
DROP_COLS = ["Market", "Date", "Modal", "Min", "Max"]  # not used as model inputs
TEST_FRACTION = 0.15  # most recent 15% of each market's history held out for validation


def load_features():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"])
    return df


def get_feature_cols(df):
    return [c for c in df.columns if c not in DROP_COLS]


def train_one_market(df, market):
    g = df[df["Market"] == market].sort_values("Date").reset_index(drop=True)
    feature_cols = get_feature_cols(g)

    X = g[feature_cols]
    y = g[TARGET]

    # Drop rows with no target, then fill remaining NaNs (early rows won't
    # have full rolling-window history yet) with column medians
    valid = y.notna()
    X, y = X[valid], y[valid]
    X = X.fillna(X.median(numeric_only=True))

    split_idx = int(len(X) * (1 - TEST_FRACTION))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if HAS_XGB:
        model = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
        )
        model_type = "XGBoost"
    else:
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        model_type = "RandomForest"

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100

    print(f"\n{market} ({model_type}, {len(X_train)} train / {len(X_test)} test rows):")
    print(f"  MAE:  Rs {mae:,.0f} per quintal")
    print(f"  MAPE: {mape:.1f}%")

    # Feature importance
    importances = getattr(model, "feature_importances_", None)
    top_features = []
    if importances is not None:
        order = np.argsort(importances)[::-1][:8]
        print("  Top features:")
        for idx in order:
            print(f"    {feature_cols[idx]}: {importances[idx]:.3f}")
            top_features.append({"feature": feature_cols[idx], "importance": float(importances[idx])})

    # Refit on ALL data (train+test) for the final deployed model, now that
    # we've measured honest validation performance above
    model.fit(X.fillna(X.median(numeric_only=True)), y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, f"model_{market.lower()}.joblib"))
    with open(os.path.join(MODELS_DIR, f"feature_cols_{market.lower()}.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(os.path.join(MODELS_DIR, f"metrics_{market.lower()}.json"), "w") as f:
        json.dump({
            "model_type": model_type,
            "mae": mae,
            "mape": mape,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "top_features": top_features,
            "trained_at": pd.Timestamp.now().isoformat(),
            "feature_medians": X.median(numeric_only=True).to_dict(),
        }, f, indent=2)

    return model, feature_cols


def main():
    if not HAS_XGB:
        print("NOTE: xgboost not installed - using RandomForest instead. "
              "Run `pip install xgboost --break-system-packages` for the stronger model.\n")

    df = load_features()
    for market in sorted(df["Market"].unique()):
        train_one_market(df, market)

    print(f"\nModels saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
