import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


VAL_SIZE = 800
TRAIN_SIZE = 3200
NUM_CLASSES = 10


def _skip_if_broken(student_code):
    if student_code.get("__error__"):
        pytest.skip("solution.ipynb failed to execute")


def _hidden_accuracy(model, ground_truth):
    X, y = ground_truth
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 100):
            out = model(X[i:i + 100])
            preds.append(out.argmax(dim=1))
    preds = torch.cat(preds)
    return (preds == y).float().mean().item()


def test_code_runs_without_error_PY1(student_code):
    """_TC1"""
    err = student_code.get("__error__")
    assert err is None, f"{type(err).__name__}: {err}"


def test_data_loading_PY2(student_code):
    """_TC2"""
    _skip_if_broken(student_code)
    loaders = [
        v for k, v in student_code.items()
        if not k.startswith("__") and isinstance(v, DataLoader)
    ]
    assert len(loaders) >= 2, (
        f"Expected at least 2 DataLoader objects (train and validation), "
        f"found {len(loaders)}."
    )
    sizes = set()
    for loader in loaders:
        try:
            sizes.add(len(loader.dataset))
        except TypeError:
            pass
    assert TRAIN_SIZE in sizes, (
        f"No DataLoader wraps a dataset of {TRAIN_SIZE} training samples. "
        f"Found dataset sizes: {sorted(sizes)}."
    )
    assert VAL_SIZE in sizes, (
        f"No DataLoader wraps a dataset of {VAL_SIZE} validation samples. "
        f"Found dataset sizes: {sorted(sizes)}."
    )


def test_lenet_architecture_PY3(student_code, lenet_model):
    """_TC3"""
    _skip_if_broken(student_code)
    assert lenet_model is not None, (
        "LeNet-5 model not found. Define it as a variable named lenet_model."
    )
    convs = [m for m in lenet_model.modules() if isinstance(m, nn.Conv2d)]
    linears = [m for m in lenet_model.modules() if isinstance(m, nn.Linear)]
    assert len(convs) >= 2, (
        f"LeNet-5 should have at least 2 convolutional layers, found {len(convs)}."
    )
    assert len(linears) >= 2, (
        f"LeNet-5 should have at least 2 fully connected layers, found {len(linears)}."
    )
    lenet_model.eval()
    with torch.no_grad():
        out = lenet_model(torch.zeros(1, 3, 32, 32))
    assert out.shape[-1] == NUM_CLASSES, (
        f"LeNet-5 output should have {NUM_CLASSES} class scores, got {out.shape[-1]}."
    )


def test_lenet_accuracy_PY4(student_code, lenet_model, ground_truth):
    """_TC4"""
    _skip_if_broken(student_code)
    if lenet_model is None:
        pytest.skip("lenet_model not found")
    acc = _hidden_accuracy(lenet_model, ground_truth)
    assert acc >= 0.40, (
        f"LeNet-5 accuracy on hidden test data = {acc:.4f}, required >= 0.40."
    )


def test_improved_model_architecture_PY5(student_code, improved_model):
    """_TC5"""
    _skip_if_broken(student_code)
    assert improved_model is not None, (
        "Improved model not found. Define it as a variable named improved_model."
    )
    has_relu = any(isinstance(m, nn.ReLU) for m in improved_model.modules())
    has_dropout = any(
        isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d))
        for m in improved_model.modules()
    )
    assert has_relu, "Improved model should use nn.ReLU activation layers."
    assert has_dropout, "Improved model should include at least one nn.Dropout layer."


def test_improved_model_accuracy_PY6(student_code, improved_model, ground_truth):
    """_TC6"""
    _skip_if_broken(student_code)
    if improved_model is None:
        pytest.skip("improved_model not found")
    acc = _hidden_accuracy(improved_model, ground_truth)
    assert acc >= 0.40, (
        f"Improved model accuracy on hidden test data = {acc:.4f}, required >= 0.40."
    )


def test_inception_architecture_PY7(student_code, inception_model):
    """_TC7"""
    _skip_if_broken(student_code)
    assert inception_model is not None, (
        "GoogLeNet-inspired model not found. "
        "Define it as a variable named inception_model."
    )
    kernels = []
    for m in inception_model.modules():
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size
            kernels.append(tuple(k) if isinstance(k, (tuple, list)) else (k, k))
    n_1x1 = sum(1 for k in kernels if k == (1, 1))
    has_3x3 = (3, 3) in kernels
    has_5x5 = (5, 5) in kernels
    has_pool = any(
        isinstance(m, nn.MaxPool2d) for m in inception_model.modules()
    )
    assert n_1x1 >= 3, (
        f"Inception module should use 1x1 convolutions in multiple branches "
        f"(found {n_1x1})."
    )
    assert has_3x3, "Inception module should include a 3x3 convolution branch."
    assert has_5x5, "Inception module should include a 5x5 convolution branch."
    assert has_pool, "Inception module should include a max pooling branch."
    inception_model.eval()
    with torch.no_grad():
        out = inception_model(torch.zeros(1, 3, 32, 32))
    assert out.shape[-1] == NUM_CLASSES, (
        f"Model output should have {NUM_CLASSES} class scores, got {out.shape[-1]}."
    )


def test_inception_accuracy_PY8(student_code, inception_model, ground_truth):
    """_TC8"""
    _skip_if_broken(student_code)
    if inception_model is None:
        pytest.skip("inception_model not found")
    acc = _hidden_accuracy(inception_model, ground_truth)
    assert acc >= 0.40, (
        f"GoogLeNet-inspired CNN accuracy on hidden test data = {acc:.4f}, "
        f"required >= 0.40."
    )


def _is_confusion_matrix(val):
    arr = None
    if isinstance(val, np.ndarray):
        arr = val
    elif isinstance(val, torch.Tensor):
        arr = val.detach().cpu().numpy()
    elif isinstance(val, pd.DataFrame):
        try:
            arr = val.to_numpy()
        except Exception:
            return False
    if arr is None or arr.shape != (NUM_CLASSES, NUM_CLASSES):
        return False
    if not np.issubdtype(arr.dtype, np.number):
        return False
    if np.any(arr < 0):
        return False
    return int(arr.sum()) == VAL_SIZE


def test_confusion_matrix_PY9(student_code):
    """_TC9"""
    _skip_if_broken(student_code)
    found = any(
        _is_confusion_matrix(v)
        for k, v in student_code.items() if not k.startswith("__")
    )
    assert found, (
        f"No {NUM_CLASSES}x{NUM_CLASSES} confusion matrix computed on the "
        f"validation set ({VAL_SIZE} samples) was found."
    )


def test_model_results_PY10(student_code):
    """_TC10"""
    _skip_if_broken(student_code)
    results = student_code.get("model_results")
    if not isinstance(results, dict):
        # Fallback: any dict with exactly 3 float accuracies in (0, 1]
        results = None
        for k, v in student_code.items():
            if k.startswith("__") or not isinstance(v, dict):
                continue
            vals = list(v.values())
            if len(vals) == 3 and all(
                isinstance(x, float) and 0.0 < x <= 1.0 for x in vals
            ):
                results = v
                break
    assert isinstance(results, dict), (
        "model_results dictionary not found. It should map the three model "
        "names to their validation accuracies."
    )
    assert len(results) == 3, (
        f"model_results should contain 3 entries, found {len(results)}."
    )
    for name, acc in results.items():
        assert isinstance(acc, float) and 0.0 < acc <= 1.0, (
            f"model_results['{name}'] should be a validation accuracy "
            f"between 0 and 1, got {acc}."
        )


def test_execution_time_PY11(student_code):
    """_TC11"""
    t = student_code.get("__exec_time__")
    assert t is not None, "Could not measure execution time."
    assert t <= 600, f"Solution took {t:.1f}s, limit is 600s."
