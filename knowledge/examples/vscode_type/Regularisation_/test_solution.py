import pytest
import numpy as np


def _skip_if_broken(student_code):
    if student_code.get("__error__"):
        pytest.skip("solution.ipynb failed")


def test_code_runs_without_error_PY1(student_code):
    """_TC1"""
    err = student_code.get("__error__")
    assert err is None, f"{type(err).__name__}: {err}"


def test_ridge_model_exists_PY2(student_code, ridge_model):
    """_TC2"""
    _skip_if_broken(student_code)
    assert ridge_model is not None, "No Ridge model found."


def test_lasso_model_exists_PY3(student_code, lasso_model):
    """_TC3"""
    _skip_if_broken(student_code)
    assert lasso_model is not None, "No Lasso model found."


def test_ridge_rmse_PY4(student_code, ridge_predictions, ground_truth):
    """_TC4"""
    _skip_if_broken(student_code)
    assert ridge_predictions is not None, "ridge_predictions not found or wrong length."
    y_true = ground_truth["Life_Expectancy"].values
    rmse = np.sqrt(np.mean((y_true - ridge_predictions) ** 2))
    assert rmse <= 5.0, f"Ridge RMSE = {rmse:.4f}, required <= 5.0."


def test_lasso_rmse_PY5(student_code, lasso_predictions, ground_truth):
    """_TC5"""
    _skip_if_broken(student_code)
    assert lasso_predictions is not None, "lasso_predictions not found or wrong length."
    y_true = ground_truth["Life_Expectancy"].values
    rmse = np.sqrt(np.mean((y_true - lasso_predictions) ** 2))
    assert rmse <= 6.0, f"Lasso RMSE = {rmse:.4f}, required <= 6.0."
