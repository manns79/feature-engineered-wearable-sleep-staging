"""Modest hyperparameter grids for train-only grouped cross-validation."""

from __future__ import annotations

LOGISTIC_PARAM_GRID = {
    "classifier__C": [0.1, 1.0, 10.0],
    "classifier__l1_ratio": [0.1, 0.5, 0.9],
}

RANDOM_FOREST_PARAM_GRID = {
    "classifier__n_estimators": [300, 500],
    "classifier__max_depth": [None, 12],
    "classifier__min_samples_leaf": [1, 5],
    "classifier__max_features": ["sqrt", 0.5],
}

XGBOOST_PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "min_child_weight": [1, 5],
}

MODEL_PARAM_GRIDS = {
    "elastic_net_logistic_regression": LOGISTIC_PARAM_GRID,
    "random_forest": RANDOM_FOREST_PARAM_GRID,
    "xgboost": XGBOOST_PARAM_GRID,
}
