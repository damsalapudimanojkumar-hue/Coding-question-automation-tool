import pytest
import numpy as np


def _skip_if_broken(student_code, model=None):
    if student_code.get("__error__"):
        pytest.skip("solution.ipynb failed")
    if model is not None and model is None:
        pytest.skip("no model found")


def test_code_runs_without_error_PY1(student_code):
    """_TC1"""
    err = student_code.get("__error__")
    assert err is None, f"{type(err).__name__}: {err}"


def test_model_exists_PY2(student_code, model):
    """_TC2"""
    _skip_if_broken(student_code)
    assert model is not None, "No LinearRegression model found."


def test_rmse_on_hidden_data_PY3(student_code, model, predictions, ground_truth):
    """_TC3"""
    _skip_if_broken(student_code, model)
    assert predictions is not None, "Could not generate predictions from model."
    y_true = ground_truth["Calories"].values
    rmse = np.sqrt(np.mean((y_true - predictions) ** 2))
    assert rmse <= 22, f"RMSE = {rmse:.2f}, required <= 22."
