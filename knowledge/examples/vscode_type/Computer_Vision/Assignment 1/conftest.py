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


# ---------- helpers: structural inspection of nn.Module ----------

def _conv_kernels(module):
    return [
        tuple(m.kernel_size) if isinstance(m.kernel_size, (tuple, list))
        else (m.kernel_size, m.kernel_size)
        for m in module.modules() if isinstance(m, nn.Conv2d)
    ]


def _has_dropout(module):
    return any(
        isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d))
        for m in module.modules()
    )


def _count_1x1(module):
    return sum(1 for k in _conv_kernels(module) if k == (1, 1))


def _outputs_10_classes(module):
    """Behavioral probe: model accepts (1, 3, 32, 32) and returns 10 logits."""
    try:
        module.eval()
        with torch.no_grad():
            out = module(torch.zeros(1, 3, 32, 32))
        return hasattr(out, "shape") and out.shape[-1] == 10
    except Exception:
        return False


def _all_classifier_models(student_code):
    models = []
    for name, val in student_code.items():
        if name.startswith("__"):
            continue
        if isinstance(val, nn.Module) and _outputs_10_classes(val):
            models.append((name, val))
    return models


def _get_named_model(student_code, var_name):
    val = student_code.get(var_name)
    if isinstance(val, nn.Module) and _outputs_10_classes(val):
        return val
    return None


@pytest.fixture(scope="session")
def lenet_model(student_code):
    if student_code.get("__error__"):
        return None
    named = _get_named_model(student_code, "lenet_model")
    if named is not None:
        return named
    # Fallback: conv net with 5x5 kernels, no 1x1 convs, no dropout
    for _, m in _all_classifier_models(student_code):
        kernels = _conv_kernels(m)
        if (kernels and (5, 5) in kernels and _count_1x1(m) == 0
                and not _has_dropout(m)):
            return m
    return None


@pytest.fixture(scope="session")
def improved_model(student_code):
    if student_code.get("__error__"):
        return None
    named = _get_named_model(student_code, "improved_model")
    if named is not None:
        return named
    # Fallback: conv net with dropout and no 1x1 convs (AlexNet-style upgrade)
    for _, m in _all_classifier_models(student_code):
        if _count_1x1(m) == 0 and _has_dropout(m):
            return m
    return None


@pytest.fixture(scope="session")
def inception_model(student_code):
    if student_code.get("__error__"):
        return None
    named = _get_named_model(student_code, "inception_model")
    if named is not None:
        return named
    # Fallback: network containing several 1x1 convs plus 3x3 and 5x5 branches
    for _, m in _all_classifier_models(student_code):
        kernels = _conv_kernels(m)
        if _count_1x1(m) >= 3 and (3, 3) in kernels and (5, 5) in kernels:
            return m
    return None


@pytest.fixture(scope="session")
def ground_truth():
    """Hidden held-out images, preprocessed the same way students are told to:
    scale to [0, 1] and convert to channel-first tensors."""
    data = np.load("tests/ground_truth.npz")
    X = torch.tensor(data["X_test"], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    y = torch.tensor(data["y_test"], dtype=torch.long)
    return X, y
