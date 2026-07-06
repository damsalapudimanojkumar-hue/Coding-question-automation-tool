import pytest
import time
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso


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
def ridge_model(student_code):
    """Detect Ridge model by isinstance across all student variables."""
    if student_code.get("__error__"):
        return None
    for name, val in student_code.items():
        if name.startswith("__"):
            continue
        if isinstance(val, Ridge):
            return val
    return None


@pytest.fixture(scope="session")
def lasso_model(student_code):
    """Detect Lasso model by isinstance across all student variables."""
    if student_code.get("__error__"):
        return None
    for name, val in student_code.items():
        if name.startswith("__"):
            continue
        if isinstance(val, Lasso):
            return val
    return None


@pytest.fixture(scope="session")
def ridge_predictions(student_code):
    """Get ridge_predictions by name, fallback to array detection."""
    if student_code.get("__error__"):
        return None
    # Try by name first
    preds = student_code.get("ridge_predictions", None)
    if preds is not None and hasattr(preds, '__len__') and len(preds) == 545:
        return np.array(preds).flatten()
    return None


@pytest.fixture(scope="session")
def lasso_predictions(student_code):
    """Get lasso_predictions by name, fallback to array detection."""
    if student_code.get("__error__"):
        return None
    # Try by name first
    preds = student_code.get("lasso_predictions", None)
    if preds is not None and hasattr(preds, '__len__') and len(preds) == 545:
        return np.array(preds).flatten()
    return None


@pytest.fixture(scope="session")
def ground_truth():
    return pd.read_csv("https://s3.ap-south-1.amazonaws.com/new-assets.ccbp.in/frontend/content/aiml/classical-ml/life_expectancy_ground_truth.csv")
