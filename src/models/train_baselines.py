"""Model factories for baseline and traditional ML sleep-stage classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


@dataclass(frozen=True)
class ModelSpec:
    """Named estimator plus metadata needed by training scripts."""

    name: str
    estimator: Any
    use_sample_weight: bool = False


def balanced_sample_weights(labels: Any) -> np.ndarray:
    """Return balanced per-row sample weights for estimators such as XGBoost."""
    return compute_sample_weight(class_weight="balanced", y=np.asarray(labels))


def majority_classifier(random_state: int = 42) -> ModelSpec:
    """Return a majority-class sanity-check classifier."""
    return ModelSpec(
        name="majority_class",
        estimator=DummyClassifier(strategy="most_frequent", random_state=random_state),
    )


def stratified_random_classifier(random_state: int = 42) -> ModelSpec:
    """Return a stratified random sanity-check classifier."""
    return ModelSpec(
        name="stratified_random",
        estimator=DummyClassifier(strategy="stratified", random_state=random_state),
    )


def elastic_net_logistic_regression(random_state: int = 42) -> ModelSpec:
    """Return multinomial elastic-net logistic regression with class weighting."""
    estimator = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    l1_ratio=0.5,
                    max_iter=5000,
                    random_state=random_state,
                    solver="saga",
                ),
            ),
        ]
    )
    return ModelSpec(name="elastic_net_logistic_regression", estimator=estimator)


def random_forest(random_state: int = 42) -> ModelSpec:
    """Return a class-weighted random forest classifier."""
    estimator = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced",
                    max_depth=None,
                    min_samples_leaf=2,
                    n_estimators=500,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return ModelSpec(name="random_forest", estimator=estimator)


def xgboost_classifier(random_state: int = 42) -> ModelSpec:
    """Return an XGBoost multiclass classifier configured for sample weights."""
    from xgboost import XGBClassifier

    return ModelSpec(
        name="xgboost",
        estimator=XGBClassifier(
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            learning_rate=0.05,
            max_depth=4,
            n_estimators=500,
            objective="multi:softprob",
            random_state=random_state,
            subsample=0.8,
            tree_method="hist",
        ),
        use_sample_weight=True,
    )


def default_model_specs(
    random_state: int = 42, include_xgboost: bool = True
) -> list[ModelSpec]:
    """Return the planned model family in project order."""
    specs = [
        majority_classifier(random_state),
        stratified_random_classifier(random_state),
        elastic_net_logistic_regression(random_state),
        random_forest(random_state),
    ]
    if include_xgboost:
        specs.append(xgboost_classifier(random_state))
    return specs
