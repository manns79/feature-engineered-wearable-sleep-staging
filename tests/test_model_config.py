from sklearn.dummy import DummyClassifier
from src.models.train_baselines import (
    default_model_specs,
    elastic_net_logistic_regression,
    random_forest,
)


def test_default_model_specs_include_two_dummy_baselines_without_xgboost():
    specs = default_model_specs(include_xgboost=False)

    assert [spec.name for spec in specs][:2] == ["majority_class", "stratified_random"]
    assert isinstance(specs[0].estimator, DummyClassifier)
    assert isinstance(specs[1].estimator, DummyClassifier)


def test_nontrivial_models_use_imputation_and_class_weighting():
    logistic = elastic_net_logistic_regression().estimator
    forest = random_forest().estimator

    assert "imputer" in logistic.named_steps
    assert logistic.named_steps["classifier"].class_weight == "balanced"
    assert "imputer" in forest.named_steps
    assert forest.named_steps["classifier"].class_weight == "balanced"
