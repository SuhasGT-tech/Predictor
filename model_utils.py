"""
Shared model-building code used by 4_train_model.py (to train/save models)
and by 5_generate_dashboard.py / 6_send_sms_forecast.py (to load them back).

WHY THIS NEEDS TO BE ITS OWN FILE
joblib (like Python's pickle underneath it) saves custom classes by
reference to the module they were defined in - not by copying the class's
code into the file. So if SeedEnsembleRegressor were defined inside
4_train_model.py, a model saved from there could only ever be loaded back
by a script that also has that exact class importable from that exact
module path. Keeping it in its own small shared module means any script
that does `import model_utils` can both create AND load these models
correctly.
"""

import numpy as np

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.ensemble import RandomForestRegressor


def make_model(params=None):
    params = params or {}
    if HAS_XGB:
        defaults = dict(n_estimators=400, max_depth=5, learning_rate=0.03,
                        subsample=0.85, colsample_bytree=0.85, random_state=42)
        defaults.update(params)
        return XGBRegressor(**defaults), "XGBoost"
    defaults = dict(n_estimators=400, max_depth=10, min_samples_leaf=3,
                    random_state=42, n_jobs=-1)
    defaults.update(params)
    return RandomForestRegressor(**defaults), "RandomForest"


class SeedEnsembleRegressor:
    """
    Trains N copies of the same model config, differing only in random
    seed, and averages their predictions. This is a standard variance-
    reduction technique (bagging over randomness) - a single tree ensemble's
    specific splits depend partly on arbitrary random choices (which rows/
    features each tree samples), so averaging several independent fits
    smooths out that arbitrary component while keeping the genuine signal.
    Exposes the same .fit()/.predict()/.feature_importances_ interface as
    a normal sklearn-style model, so nothing downstream needs to change.
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
            model, _ = make_model(p)
            model.fit(X, y, sample_weight=sample_weight)
            self.models.append(model)
        return self

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds.mean(axis=1)

    @property
    def feature_importances_(self):
        return np.mean([m.feature_importances_ for m in self.models], axis=0)
