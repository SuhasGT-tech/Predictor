"""
Shared model-building code used by 4_train_model.py (to train/save models)
and by 5_generate_dashboard.py / 6_send_sms_forecast.py (to load them back).

WHY THIS NEEDS TO BE ITS OWN FILE
joblib (like Python's pickle underneath it) saves custom classes by
reference to the module they were defined in - not by copying the class's
code into the file. Keeping it in its own small shared module means any
script that does `import model_utils` can both create AND load these models
correctly.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# Core trend & momentum features used by the regularized linear model
CORE_TREND_FEATS = [
    "ret_lag_1", "ret_lag_2", "ret_sum_4", "ret_sum_12",
    "trend_gap_12", "spread_other", "other_ret_4d", "is_festival_window"
]


class BlendRegressor:
    """
    Robust ensemble for commodity tender price prediction:
    1. Regularized linear pipeline (StandardScaler + Ridge) on core trend,
       momentum, cross-market spread, and festival signals. This provides
       strong shrinkage and eliminates overfitting on noisy financial returns.
    2. Regularized Tree Ensemble (XGBoost with shallow depth and child weight
       or RandomForest) capturing non-linear interactions across all stationary features.
    3. Blended prediction combining the two components with target winsorization.
    """
    def __init__(self, ridge_alpha=300.0, blend_weight=0.5, tree_params=None):
        self.ridge_alpha = ridge_alpha
        self.blend_weight = blend_weight
        self.tree_params = tree_params or {}
        
        self.ridge = make_pipeline(StandardScaler(), Ridge(alpha=self.ridge_alpha))
        if HAS_XGB:
            defaults = dict(
                n_estimators=150, max_depth=2, learning_rate=0.03,
                subsample=0.75, colsample_bytree=0.75, min_child_weight=15,
                reg_lambda=10.0, random_state=42
            )
            defaults.update(self.tree_params)
            self.tree = XGBRegressor(**defaults)
        else:
            defaults = dict(
                n_estimators=200, max_depth=4, min_samples_leaf=5,
                random_state=42, n_jobs=-1
            )
            defaults.update(self.tree_params)
            self.tree = RandomForestRegressor(**defaults)

        self.feature_names_in_ = None
        self.core_feats = None

    def fit(self, X, y, sample_weight=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            self.core_feats = [c for c in CORE_TREND_FEATS if c in X.columns]
            if not self.core_feats:
                self.core_feats = self.feature_names_in_
            X_df = X
        else:
            X_df = pd.DataFrame(X)
            self.feature_names_in_ = list(range(X_df.shape[1]))
            self.core_feats = self.feature_names_in_

        # Winsorize target returns to [1st, 99th] percentiles to protect
        # against recording typos / extreme outlier distortion
        y_arr = np.asarray(y)
        q_low, q_high = np.percentile(y_arr, 1), np.percentile(y_arr, 99)
        y_clipped = np.clip(y_arr, q_low, q_high)

        self.ridge.fit(X_df[self.core_feats], y_clipped)
        if sample_weight is not None:
            self.tree.fit(X_df, y_clipped, sample_weight=sample_weight)
        else:
            self.tree.fit(X_df, y_clipped)
        return self

    def predict(self, X):
        if hasattr(X, "columns"):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        pr_ridge = self.ridge.predict(X_df[self.core_feats])
        pr_tree = self.tree.predict(X_df)
        return self.blend_weight * pr_ridge + (1.0 - self.blend_weight) * pr_tree

    @property
    def feature_importances_(self):
        if hasattr(self.tree, "feature_importances_"):
            return self.tree.feature_importances_
        return np.ones(len(self.feature_names_in_)) / len(self.feature_names_in_)


class SeedEnsembleRegressor:
    """
    Kept for backward compatibility with previous saved models.
    """
    def __init__(self, params=None, n_seeds=5, base_seed=42):
        self.params = params or {}
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.models = []

    def fit(self, X, y, sample_weight=None):
        self.models = []
        for i in range(self.n_seeds):
            p = dict(self.params)
            p["random_state"] = self.base_seed + i
            if HAS_XGB:
                defaults = dict(n_estimators=150, max_depth=3, learning_rate=0.03,
                                subsample=0.8, colsample_bytree=0.8)
                defaults.update(p)
                m = XGBRegressor(**defaults)
            else:
                defaults = dict(n_estimators=200, max_depth=5, min_samples_leaf=3, n_jobs=-1)
                defaults.update(p)
                m = RandomForestRegressor(**defaults)
            m.fit(X, y, sample_weight=sample_weight)
            self.models.append(m)
        return self

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds.mean(axis=1)

    @property
    def feature_importances_(self):
        return np.mean([m.feature_importances_ for m in self.models], axis=0)


def make_model(params=None):
    model_type = "Ensemble (Ridge + XGBoost)" if HAS_XGB else "Ensemble (Ridge + RandomForest)"
    return BlendRegressor(tree_params=params), model_type
