"""
Train a price-prediction model per market (Arasikere, Tiptur) using the
feature table from step 3.

KEY CHANGE FROM THE FIRST VERSION: predicting CHANGE, not the price level
--------------------------------------------------------------------------
Tree-based models (XGBoost/RandomForest) predict by averaging training
examples in a leaf - they physically CANNOT output a number higher than
the highest price ever seen in training, or lower than the lowest. Since
copra prices have climbed for 24 years (~Rs 3,000 in 2002 to Rs 31,000+
now), a model trained directly on price level is quietly capped near
whatever the highest price it was trained on happens to be - it can't
"see" further upside even if every signal points that way.

Fix: predict the LOG RETURN (log(next_price / last_price)) instead - a
value that's stayed in a stable, bounded range historically regardless of
what the absolute price level is. We then reconstruct the actual price as:
    predicted_price = last_known_price * exp(predicted_log_return)
This lets the forecast extrapolate to price levels never seen in training,
because it's the RATIO that's being predicted, not the absolute number.

WHY WALK-FORWARD CROSS-VALIDATION, NOT ONE TRAIN/TEST SPLIT
A single holdout at the end of history gives you one noisy estimate of
accuracy - if that particular stretch happened to be unusually calm or
volatile, your accuracy number is misleading. Walk-forward CV trains on
an expanding window and tests on each subsequent chunk in turn (still
always training on the past, testing on the future - never shuffled),
giving a more honest average accuracy across several different periods.

OUTPUT
------
  models/model_<market>.joblib       - trained model (predicts log-return)
  models/feature_cols_<market>.json  - exact feature list/order the model expects
  models/metrics_<market>.json       - validation performance + residual std
                                        (used for prediction interval width)

USAGE
-----
    python 4_train_model.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from model_utils import make_model, SeedEnsembleRegressor, HAS_XGB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(BASE_DIR, "data", "features.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DROP_COLS = ["Market", "Date", "Modal", "Min", "Max"]  # not used as model inputs
N_CV_FOLDS = 5
MIN_TRAIN_FRACTION = 0.5  # first fold trains on at least this much of history


def load_features():
    return pd.read_csv(FEATURES_CSV, parse_dates=["Date"])


def get_feature_cols(df):
    return [c for c in df.columns if c not in DROP_COLS]


# A handful of candidate configs to try per market - kept small so this stays
# fast, but covers the main tradeoffs (tree depth/count vs. overfitting risk)
XGB_CANDIDATES = [
    {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.03},
    {"n_estimators": 600, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.02},
    # Regularized variants - test whether penalizing complexity helps
    # generalization, given the model leans heavily on a handful of features
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
     "reg_alpha": 0.5, "reg_lambda": 2.0, "min_child_weight": 5},
    {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.02,
     "reg_alpha": 1.0, "reg_lambda": 3.0, "min_child_weight": 8},
]
RF_CANDIDATES = [
    {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 3},
    {"n_estimators": 600, "max_depth": 6, "min_samples_leaf": 5},
    {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 2},
    # More regularized (higher min_samples_leaf = each leaf must average
    # over more rows, a similar generalization-vs-complexity tradeoff)
    {"n_estimators": 500, "max_depth": 5, "min_samples_leaf": 10},
]


def pick_best_hyperparams(X, y_fit, y_eval, lag1, weights=None):
    """
    Try each candidate config, score it with the same walk-forward CV used
    for final reporting, and keep whichever scored the lowest average MAPE.
    This replaces guessing hyperparameters up front with actually testing
    a few reasonable options against your specific data.

    Also tests WITH and WITHOUT recency weighting for each config, since
    whether older data is still representative of current market conditions
    is itself an empirical question, not something to assume either way.
    """
    candidates = XGB_CANDIDATES if HAS_XGB else RF_CANDIDATES
    best_params, best_mape, best_results, best_use_weights = None, np.inf, None, False

    for params in candidates:
        for use_weights, label in [(False, "no recency weighting"), (True, "recency-weighted")]:
            w = weights if use_weights else None
            results = walk_forward_cv(X, y_fit, y_eval, lag1, weights=w, params=params)
            avg_mape = float(np.mean([r["mape"] for r in results])) if results else np.inf
            print(f"    trying {params} ({label}) -> avg MAPE {avg_mape:.1f}%")
            if avg_mape < best_mape:
                best_params, best_mape, best_results, best_use_weights = params, avg_mape, results, use_weights

    print(f"  -> best used {'recency weighting' if best_use_weights else 'no recency weighting'}")
    return best_params, best_results, best_use_weights


def walk_forward_cv(X, y_fit, y_eval, lag1, weights=None, n_folds=N_CV_FOLDS,
                     min_train_frac=MIN_TRAIN_FRACTION, params=None, n_seeds=1):
    """
    Expanding-window walk-forward validation. Returns list of per-fold
    (mae_price, mape_price) reconstructed in actual Rs terms, not log-return
    units, so the reported accuracy is directly interpretable.

    y_fit is what the model is TRAINED on (winsorized, so a handful of
    likely data-entry-error outliers don't distort what it learns).
    y_eval is the TRUE, unclipped outcome, used only to score predictions -
    so accuracy numbers reflect real-world performance, not a smoothed
    target the model was never really tested against.

    weights (optional): per-row sample weights (e.g. recency weighting -
    more recent tenders count more than very old ones during training).
    """
    n = len(X)
    start = int(n * min_train_frac)
    fold_edges = np.linspace(start, n, n_folds + 1).astype(int)

    results = []
    for i in range(n_folds):
        train_end = fold_edges[i]
        test_end = fold_edges[i + 1]
        if train_end <= 10 or test_end <= train_end:
            continue

        X_train, y_train = X.iloc[:train_end], y_fit.iloc[:train_end]
        X_test, y_test_true = X.iloc[train_end:test_end], y_eval.iloc[train_end:test_end]
        lag1_test = lag1.iloc[train_end:test_end]
        w_train = weights.iloc[:train_end].values if weights is not None else None

        model, _ = make_model(params) if n_seeds <= 1 else (SeedEnsembleRegressor(params, n_seeds=n_seeds), None)
        model.fit(X_train, y_train, sample_weight=w_train)
        pred_log_return = model.predict(X_test)

        pred_price = lag1_test.values * np.exp(pred_log_return)
        actual_price = lag1_test.values * np.exp(y_test_true.values)

        mae = float(np.mean(np.abs(pred_price - actual_price)))
        mape = float(np.mean(np.abs((pred_price - actual_price) / actual_price)) * 100)
        results.append({"mae": mae, "mape": mape, "test_rows": len(X_test)})

    return results


def train_one_market(df, market):
    g = df[df["Market"] == market].sort_values("Date").reset_index(drop=True)
    feature_cols = get_feature_cols(g)

    # Target: log-return relative to the previous tender's actual price.
    # Rows with no modal_lag_1 (the very first couple of tenders on record)
    # have no defined target and are dropped.
    valid = g["modal_lag_1"].notna() & (g["modal_lag_1"] > 0) & g["Modal"].notna() & (g["Modal"] > 0)
    g = g[valid].reset_index(drop=True)
    y_raw = np.log(g["Modal"] / g["modal_lag_1"])
    lag1 = g["modal_lag_1"]

    # Winsorize extreme log-returns before training: a single day where the
    # portal recorded (or later corrected) a wildly implausible price swing
    # would otherwise dominate what the model learns, since squared-error-style
    # loss functions weight big errors heavily. Clipping at the 1st/99th
    # percentile keeps those rows in the data (so we don't lose real signal)
    # while stopping them from distorting training. Evaluation below still
    # compares against the true, unclipped outcome - only the training target
    # is smoothed.
    lo, hi = y_raw.quantile(0.01), y_raw.quantile(0.99)
    n_clipped = int(((y_raw < lo) | (y_raw > hi)).sum())
    y = y_raw.clip(lo, hi)
    if n_clipped:
        print(f"  Winsorized {n_clipped} extreme log-return outlier(s) "
              f"(likely data-entry errors) before training")

    X = g[feature_cols].fillna(g[feature_cols].median(numeric_only=True))

    # Recency weights: older tenders count less than recent ones, since a
    # 24-year-old market regime may not reflect current dynamics. Half-life
    # of 4 years means data from 4 years ago carries half the weight of
    # today's; 8 years ago, a quarter; etc. Whether this actually helps is
    # tested empirically below (alongside hyperparameters), not assumed.
    age_years = (g["Date"].max() - g["Date"]).dt.days / 365.25
    HALF_LIFE_YEARS = 4.0
    recency_weights = 0.5 ** (age_years / HALF_LIFE_YEARS)

    print(f"\n{market}: searching hyperparameters ({N_CV_FOLDS}-fold walk-forward CV each)...")
    best_params, cv_results, use_weights = pick_best_hyperparams(
        X, y, y_raw, lag1, weights=recency_weights)
    avg_mae = float(np.mean([r["mae"] for r in cv_results])) if cv_results else None
    avg_mape = float(np.mean([r["mape"] for r in cv_results])) if cv_results else None

    model_type = "XGBoost" if HAS_XGB else "RandomForest"
    print(f"  Best config: {best_params} (recency weighting: {use_weights})")
    print(f"  Walk-forward avg over {len(cv_results)} folds: "
          f"MAE Rs {avg_mae:,.0f}, MAPE {avg_mape:.1f}%")
    for i, r in enumerate(cv_results):
        print(f"    fold {i+1}: MAE Rs {r['mae']:,.0f}, MAPE {r['mape']:.1f}% ({r['test_rows']} rows)")

    final_weights = recency_weights if use_weights else None

    # Test whether bagging N differently-seeded copies of the chosen config
    # and averaging their predictions beats a single model - a standard
    # variance-reduction technique, but checked empirically here rather than
    # assumed, using the same walk-forward setup for a fair comparison.
    N_ENSEMBLE_SEEDS = 5
    ensemble_results = walk_forward_cv(X, y, y_raw, lag1, weights=final_weights,
                                        params=best_params, n_seeds=N_ENSEMBLE_SEEDS)
    ensemble_mape = float(np.mean([r["mape"] for r in ensemble_results])) if ensemble_results else np.inf
    use_ensemble = ensemble_mape < avg_mape
    print(f"  Single model MAPE: {avg_mape:.1f}% vs {N_ENSEMBLE_SEEDS}-seed ensemble MAPE: "
          f"{ensemble_mape:.1f}% -> using {'ensemble' if use_ensemble else 'single model'}")
    if use_ensemble:
        cv_results = ensemble_results
        avg_mae = float(np.mean([r["mae"] for r in cv_results]))
        avg_mape = ensemble_mape

    n_seeds_final = N_ENSEMBLE_SEEDS if use_ensemble else 1

    # Residual std (in log-return units) from the LAST fold's out-of-sample
    # predictions, scored against the TRUE (unclipped) outcome - used later
    # to build a rough +/- price range, since a single point estimate
    # overstates how precise this really is
    residual_std = None
    if cv_results:
        last_train_end = int(np.linspace(int(len(X) * MIN_TRAIN_FRACTION), len(X), N_CV_FOLDS + 1).astype(int)[-2])
        model = SeedEnsembleRegressor(best_params, n_seeds=n_seeds_final) if use_ensemble else make_model(best_params)[0]
        w_fit = final_weights.iloc[:last_train_end].values if final_weights is not None else None
        model.fit(X.iloc[:last_train_end], y.iloc[:last_train_end], sample_weight=w_fit)
        resid = y_raw.iloc[last_train_end:].values - model.predict(X.iloc[last_train_end:])
        if len(resid) > 1:
            residual_std = float(np.std(resid))

    # Final model: refit on ALL available data for deployment, now that
    # we've measured honest out-of-sample accuracy above
    final_model = SeedEnsembleRegressor(best_params, n_seeds=n_seeds_final) if use_ensemble else make_model(best_params)[0]
    final_model.fit(X, y, sample_weight=final_weights.values if final_weights is not None else None)

    importances = getattr(final_model, "feature_importances_", None)
    top_features = []
    if importances is not None:
        order = np.argsort(importances)[::-1][:8]
        print("  Top features (final model):")
        for idx in order:
            print(f"    {feature_cols[idx]}: {importances[idx]:.3f}")
            top_features.append({"feature": feature_cols[idx], "importance": float(importances[idx])})

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(final_model, os.path.join(MODELS_DIR, f"model_{market.lower()}.joblib"))
    with open(os.path.join(MODELS_DIR, f"feature_cols_{market.lower()}.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(os.path.join(MODELS_DIR, f"metrics_{market.lower()}.json"), "w") as f:
        json.dump({
            "model_type": model_type,
            "target": "log_return",
            "hyperparameters": best_params,
            "recency_weighted": use_weights,
            "seed_ensembled": use_ensemble,
            "n_seeds": n_seeds_final,
            "mae": avg_mae,
            "mape": avg_mape,
            "residual_std_log_return": residual_std,
            "cv_folds": cv_results,
            "train_rows": len(X),
            "top_features": top_features,
            "trained_at": pd.Timestamp.now().isoformat(),
            "feature_medians": X.median(numeric_only=True).to_dict(),
        }, f, indent=2)

    return final_model, feature_cols


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
