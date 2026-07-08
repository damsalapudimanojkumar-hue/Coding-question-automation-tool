import pytest
import torch
import torch.nn as nn
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
    assert model is not None, "No nn.Module instance found."


def test_multi_scale_convolutions_PY3(student_code, model, model_conv_kernel_sizes):
    """_TC3"""
    _skip_if_broken(student_code, model)
    num_distinct = len(model_conv_kernel_sizes)
    assert num_distinct >= 3, (
        f"Found {num_distinct} distinct Conv2d kernel size(s): {model_conv_kernel_sizes}. "
        f"An Inception module needs at least 3 (e.g. 1x1, 3x3, 5x5)."
    )


def test_bottleneck_1x1_convolutions_PY4(student_code, model, model_1x1_conv_count):
    """_TC4"""
    _skip_if_broken(student_code, model)
    assert model_1x1_conv_count >= 3, (
        f"Found {model_1x1_conv_count} Conv2d layer(s) with kernel_size=1. "
        f"An Inception module typically needs at least 3 for bottleneck reduction."
    )


def test_output_shape_PY5(student_code, model):
    """_TC5"""
    _skip_if_broken(student_code, model)
    model.eval()
    sample = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        try:
            out = model(sample)
        except Exception as e:
            pytest.fail(f"Model failed on (1, 3, 32, 32) input: {e}")
    assert out.shape[-1] == 10, (
        f"Output has {out.shape[-1]} units, expected 10 for 10-class classification."
    )


def test_accuracy_PY6(student_code, model, ground_truth):
    """_TC6"""
    _skip_if_broken(student_code, model)
    X_test, y_test = ground_truth
    model.eval()
    X_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2) / 255.0
    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
    accuracy = (preds.numpy() == y_test).mean()
    assert accuracy >= 0.70, (
        f"Accuracy = {accuracy:.4f}, required >= 0.70."
    )
