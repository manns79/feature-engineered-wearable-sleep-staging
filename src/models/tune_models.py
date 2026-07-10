"""Small hyperparameter grids for validation-set model selection."""

from __future__ import annotations

LOGISTIC_PARAM_GRID = {
    "classifier__C": [0.01, 0.1, 1.0, 10.0],
    "classifier__l1_ratio": [0.1, 0.5, 0.9],
}

RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [None, 8, 16],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", 0.5],
}

XGBOOST_PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
