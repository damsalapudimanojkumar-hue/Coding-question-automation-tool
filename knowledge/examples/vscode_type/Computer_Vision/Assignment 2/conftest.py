import pytest
import time
import json
import numpy as np
import torch
import torch.nn as nn


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
    """Find the trained nn.Module (pick the one with the most parameters)."""
    if student_code.get("__error__"):
        return None
    best_model = None
    best_params = 0
    for name, val in student_code.items():
        if name.startswith("__"):
            continue
        if isinstance(val, nn.Module):
            num_params = sum(p.numel() for p in val.parameters())
            if num_params > best_params:
                best_params = num_params
                best_model = val
    return best_model


@pytest.fixture(scope="session")
def model_conv_kernel_sizes(model):
    """Collect all unique Conv2d kernel sizes from the model."""
    if model is None:
        return set()
    sizes = set()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            sizes.add(m.kernel_size)
    return sizes


@pytest.fixture(scope="session")
def model_1x1_conv_count(model):
    """Count Conv2d layers with kernel_size=1 (bottleneck convolutions)."""
    if model is None:
        return 0
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
            count += 1
    return count


@pytest.fixture(scope="session")
def ground_truth():
    """Load hidden test data for accuracy evaluation."""
    gt = np.load("tests/ground_truth.npz")
    return gt["X_test"], gt["y_test"]
