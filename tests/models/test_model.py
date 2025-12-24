import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.model import build_logestic_model, build_rf_model


def test_build_logestic_model_returns_logistic_regression():
    """
    Test that build_logestic_model returns a LogisticRegression instance
    with the expected hyperparameters.
    """
    model = build_logestic_model()
    assert isinstance(model, LogisticRegression)
    assert model.max_iter == 1000
    assert model.solver == "liblinear"
    assert model.random_state == 42


def test_build_rf_model_returns_random_forest():
    """
    Test that build_rf_model returns a RandomForestClassifier instance
    with the expected hyperparameters.
    """
    model = build_rf_model()
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 200
    assert model.max_depth == 10
    assert model.min_samples_split == 5
    assert model.random_state == 42
    assert model.n_jobs == -1
