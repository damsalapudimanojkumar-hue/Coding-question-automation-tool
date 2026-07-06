import pytest
import time
import json
import pandas as pd
import numpy as np


def _load_notebook_code(path):
    """Parse .ipynb JSON and return concatenated source of all code cells."""
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    chunks = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, list):
            src = "".join(src)
        cleaned = "\n".join(
            line for line in src.splitlines()
            if not line.lstrip().startswith(("%", "!", "?"))
        )
        chunks.append(cleaned)
    return "\n\n".join(chunks)


@pytest.fixture(scope="session")
def student_code():
    student_vars = {"__error__": None, "__exec_time__": None}
    start = time.time()
    try:
        code = _load_notebook_code("solution.ipynb")
        exec(code, student_vars)
    except Exception as e:
        student_vars["__error__"] = e
    student_vars["__exec_time__"] = time.time() - start
    return student_vars


@pytest.fixture(scope="session")
def model(student_code):
    """Find sklearn LinearRegression model by type."""
    if student_code.get("__error__"):
        return None
    from sklearn.linear_model import LinearRegression
    for name, val in student_code.items():
        if name.startswith("__"):
            continue
        if isinstance(val, LinearRegression):
            return val
    return None


@pytest.fixture(scope="session")
def ground_truth():
    return pd.read_csv("https://s3.ap-south-1.amazonaws.com/new-assets.ccbp.in/frontend/content/aiml/classical-ml/slr_ground_truth.csv")


@pytest.fixture(scope="session")
def predictions(student_code, model, ground_truth):
    """Use student's model to predict on hidden test data."""
    if model is None:
        return None
    try:
        X_test = ground_truth[["Duration"]]
        y_pred = model.predict(X_test)
        return y_pred
    except Exception:
        return None
