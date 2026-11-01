"""
Train a price-prediction model per market (Arasikere, Tiptur) using the
clean stationary feature table from step 3.

KEY DESIGN PRINCIPLES:
1. Stationarity & Scale Invariance:
   Target is log-return between tenders: log(Modal / modal_lag_1).
   Input features are log returns, rolling momentum sums, trend gaps,
   cross-market spreads, and macroeconomic changes. Tree models and linear
   models generalize across all price levels without out-of-range distortions.

2. Regularized Blend Architecture:
   Combines a shrinkage linear model (StandardScaler + Ridge) capturing persistent
   momentum and mean-reversion trends, with a shallow regularized tree ensemble
   (XGBoost / RandomForest) capturing non-linear interactions.

3. Walk-Forward Cross-Validation:
   Honest out-of-sample evaluation across expanding time windows (never shuffled),
   measuring MAE and MAPE in actual Rupee terms.

OUTPUT:
  models/model_<market>.joblib       - trained model
  models/feature_cols_<market>.json  - exact feature list/order the model expects
  models/metrics_<market>.json       - validation performance + residual std
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from model_utils import make_model, BlendRegressor, HAS_XGB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(BASE_DIR, "data", "features.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Exclude raw absolute price levels, metadata, and raw dates from model inputs
EXCLUDE_COLS = [
    "Market", "Date", "Modal", "Min", "Max", "Arrivals", "target",
    "last_price", "modal_lag_1", "modal_lag_2", "modal_lag_3", "modal_lag_4", "modal_lag_5",
    "other_market_last_price", "year", "days_since_start", "day_of_week", "month",
    "is_tiptur_tenderday", "is_arsikere_tenderday", "arrivals_lag_1", "arrivals_roll_mean_4",
    "coconut_oil_price_usd_per_mt", "coconut_oil_price_change_1m", "usd_inr_rate",
    "rainfall_mm", "temp_c", "rainfall_roll_90"
]

N_CV_FOLDS = 5
MIN_TRAIN_FRACTION = 0.5


def load_features():
    return pd.read_csv(FEATURES_CSV, parse_dates=["Date"])


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS and not c.startswith("rain_") or c in ["rain_90d", "rain_30d"]]


CANDIDATE_PARAMS = [
    {"tree_params": {"max_depth": 2, "n_estimators": 120, "learning_rate": 0.03, "min_child_weight": 15}, "ridge_alpha": 300.0, "blend_weight": 0.5},
    {"tree_params": {"max_depth": 2, "n_estimators": 160, "learning_rate": 0.02, "min_child_weight": 20}, "ridge_alpha": 200.0, "blend_weight": 0.5},
    {"tree_params": {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.03, "min_child_weight": 25}, "ridge_alpha": 400.0, "blend_weight": 0.4},
]


def walk_forward_cv(X, y, last_price, n_folds=N_CV_FOLDS, min_train_frac=MIN_TRAIN_FRACTION, config=None):
    """
    Expanding-window walk-forward validation. Reconstructs actual prices
    in Rs terms to report true out-of-sample MAE, MAPE, and directional accuracy.
    """
    n = len(X)
    start = int(n * min_train_frac)
    fold_edges = np.linspace(start, n, n_folds + 1).astype(int)

    results = []
    for i in range(n_folds):
        train_end = fold_edges[i]
        test_end = fold_edges[i + 1]
        if train_end <= 20 or test_end <= train_end:
            continue

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
        last_price_test = last_price.iloc[train_end:test_end]

        model = BlendRegressor(
            ridge_alpha=config.get("ridge_alpha", 300.0),
            blend_weight=config.get("blend_weight", 0.5),
            tree_params=config.get("tree_params", {})
        )
        model.fit(X_train, y_train)
        pred_log_return = model.predict(X_test)

        pred_price = last_price_test.values * np.exp(pred_log_return)
        actual_price = last_price_test.values * np.exp(y_test.values)

        mae = float(np.mean(np.abs(pred_price - actual_price)))
        mape = float(np.mean(np.abs((pred_price - actual_price) / actual_price)) * 100)
        
        # Direction accuracy on non-zero movements
        nz = y_test.values != 0
        dir_acc = float((np.sign(pred_log_return[nz]) == np.sign(y_test.values[nz])).mean() * 100) if np.any(nz) else 50.0

        results.append({
            "mae": mae,
            "mape": mape,
            "dir_acc": dir_acc,
            "test_rows": len(X_test)
        })

    return results


def pick_best_config(X, y, last_price):
    best_config, best_mape, best_results = None, np.inf, None

    for config in CANDIDATE_PARAMS:
        results = walk_forward_cv(X, y, last_price, config=config)
        avg_mape = float(np.mean([r["mape"] for r in results])) if results else np.inf
        print(f"    config: tree_depth={config['tree_params']['max_depth']}, ridge_alpha={config['ridge_alpha']} -> avg MAPE {avg_mape:.2f}%")
        if avg_mape < best_mape:
            best_config, best_mape, best_results = config, avg_mape, results

    return best_config, best_results


def train_one_market(df, market):
    g = df[df["Market"] == market].sort_values("Date").reset_index(drop=True)
    feature_cols = [c for c in get_feature_cols(g) if c in g.columns]

    valid = (
        g["modal_lag_1"].notna() & (g["modal_lag_1"] > 0) &
        g["Modal"].notna() & (g["Modal"] > 0) &
        g["target"].notna()
    )
    g = g[valid].reset_index(drop=True)

    X = g[feature_cols].copy()
    medians = X.median(numeric_only=True).to_dict()
    X = X.fillna(medians)

    y = g["target"]
    last_price = g["modal_lag_1"]

    print(f"\n{market}: searching optimal model configuration ({N_CV_FOLDS}-fold walk-forward CV)...")
    best_config, cv_results = pick_best_config(X, y, last_price)

    avg_mae = float(np.mean([r["mae"] for r in cv_results])) if cv_results else None
    avg_mape = float(np.mean([r["mape"] for r in cv_results])) if cv_results else None
    avg_dir = float(np.mean([r["dir_acc"] for r in cv_results])) if cv_results else None

    print(f"  Best config: {best_config}")
    print(f"  Walk-forward avg over {len(cv_results)} folds: "
          f"MAE Rs {avg_mae:,.0f}, MAPE {avg_mape:.2f}%, Direction Accuracy {avg_dir:.1f}%")
    for i, r in enumerate(cv_results):
        print(f"    fold {i+1}: MAE Rs {r['mae']:,.0f}, MAPE {r['mape']:.2f}%, Dir {r['dir_acc']:.1f}% ({r['test_rows']} rows)")

    # Residual std from out-of-sample predictions
    residual_std = None
    if cv_results:
        last_train_end = int(np.linspace(int(len(X) * MIN_TRAIN_FRACTION), len(X), N_CV_FOLDS + 1).astype(int)[-2])
        m_eval = BlendRegressor(
            ridge_alpha=best_config.get("ridge_alpha", 300.0),
            blend_weight=best_config.get("blend_weight", 0.5),
            tree_params=best_config.get("tree_params", {})
        )
        m_eval.fit(X.iloc[:last_train_end], y.iloc[:last_train_end])
        resid = y.iloc[last_train_end:].values - m_eval.predict(X.iloc[last_train_end:])
        if len(resid) > 1:
            residual_std = float(np.std(resid))

    # Final deployment model trained on full available history
    final_model = BlendRegressor(
        ridge_alpha=best_config.get("ridge_alpha", 300.0),
        blend_weight=best_config.get("blend_weight", 0.5),
        tree_params=best_config.get("tree_params", {})
    )
    final_model.fit(X, y)

    importances = getattr(final_model, "feature_importances_", None)
    top_features = []
    if importances is not None:
        order = np.argsort(importances)[::-1][:8]
        print("  Top features (final tree model):")
        for idx in order:
            feat_name = feature_cols[idx]
            imp_val = float(importances[idx])
            print(f"    {feat_name}: {imp_val:.3f}")
            top_features.append({"feature": feat_name, "importance": imp_val})

    model_label = "Ensemble (Ridge + XGBoost)" if HAS_XGB else "Ensemble (Ridge + RandomForest)"

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(final_model, os.path.join(MODELS_DIR, f"model_{market.lower()}.joblib"))
    with open(os.path.join(MODELS_DIR, f"feature_cols_{market.lower()}.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(os.path.join(MODELS_DIR, f"metrics_{market.lower()}.json"), "w") as f:
        json.dump({
            "model_type": model_label,
            "target": "log_return",
            "hyperparameters": best_config,
            "mae": avg_mae,
            "mape": avg_mape,
            "direction_accuracy": avg_dir,
            "residual_std_log_return": residual_std,
            "cv_folds": cv_results,
            "train_rows": len(X),
            "top_features": top_features,
            "trained_at": pd.Timestamp.now().isoformat(),
            "feature_medians": medians,
        }, f, indent=2)

    return final_model, feature_cols


def main():
    df = load_features()
    for market in sorted(df["Market"].unique()):
        train_one_market(df, market)

    print(f"\nAll models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
