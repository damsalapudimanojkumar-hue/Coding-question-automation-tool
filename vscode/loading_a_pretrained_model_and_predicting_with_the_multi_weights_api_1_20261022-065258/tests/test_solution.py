import pytest
import torch
from torchvision.models import EfficientNet


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _skip_if_broken(student_code):
    if student_code.get("__error__"):
        pytest.skip("solution.ipynb raised an exception during execution")


# ---------------------------------------------------------------------------
# PY1: Notebook runs without error
# ---------------------------------------------------------------------------

def test_code_runs_without_error_PY1(student_code):
    """Notebook must execute end-to-end with no exceptions."""
    err = student_code.get("__error__")
    assert err is None, f"{type(err).__name__}: {err}"


# ---------------------------------------------------------------------------
# PY2: EfficientNet model detected and in eval mode
# ---------------------------------------------------------------------------

def test_efficientnet_in_eval_mode_PY2(student_code, student_model):
    """EfficientNet model must exist and have eval() called on it."""
    _skip_if_broken(student_code)
    assert student_model is not None, (
        "No EfficientNet model found in the notebook namespace. "
        "Make sure to build it with efficientnet_b0(weights=...) and assign it to 'model'."
    )
    assert not student_model.training, (
        "Model is still in training mode. Call model.eval() before running inference."
    )


# ---------------------------------------------------------------------------
# PY3: predictions is a list of 210 valid ImageNet class-name strings
# ---------------------------------------------------------------------------

def test_predictions_valid_imagenet_strings_PY3(student_code, student_predictions, valid_categories):
    """predictions must be a list of 210 strings, all valid ImageNet category names."""
    _skip_if_broken(student_code)
    assert student_predictions is not None, (
        "No 'predictions' list found. Store your ImageNet class-name strings in a variable called 'predictions'."
    )
    assert isinstance(student_predictions, list), (
        f"'predictions' must be a Python list, got {type(student_predictions).__name__}."
    )
    assert len(student_predictions) == 210, (
        f"Expected 210 predictions (one per student image), got {len(student_predictions)}."
    )
    invalid = [p for p in student_predictions if p not in valid_categories]
    assert len(invalid) == 0, (
        f"Found {len(invalid)} prediction(s) that are not valid ImageNet category names. "
        f"Examples: {invalid[:5]}. Use weights.meta['categories'] to map indices to names."
    )


# ---------------------------------------------------------------------------
# PY4: classifier_in_features == 1280 and classifier_out_features == 1000
# ---------------------------------------------------------------------------

def test_classifier_features_PY4(student_code):
    """classifier_in_features and classifier_out_features must be correct integers."""
    _skip_if_broken(student_code)

    in_feat = student_code.get("classifier_in_features", None)
    out_feat = student_code.get("classifier_out_features", None)

    assert in_feat is not None, (
        "'classifier_in_features' variable not found. "
        "Read it from model.classifier's Linear layer."
    )
    assert out_feat is not None, (
        "'classifier_out_features' variable not found. "
        "Read it from model.classifier's Linear layer."
    )
    assert isinstance(in_feat, int), (
        f"'classifier_in_features' must be a plain Python int, got {type(in_feat).__name__}."
    )
    assert isinstance(out_feat, int), (
        f"'classifier_out_features' must be a plain Python int, got {type(out_feat).__name__}."
    )
    assert in_feat == 1280, (
        f"classifier_in_features should be 1280, got {in_feat}. "
        "Inspect the Linear layer inside model.classifier."
    )
    assert out_feat == 1000, (
        f"classifier_out_features should be 1000, got {out_feat}. "
        "Inspect the Linear layer inside model.classifier."
    )


# ---------------------------------------------------------------------------
# PY5: Hidden inference -- 90 valid predictions with >= 2 unique classes
# ---------------------------------------------------------------------------

def test_hidden_inference_PY5(student_code, student_model, hidden_inference, valid_categories):
    """Model must produce 90 valid ImageNet predictions on the hidden test images."""
    _skip_if_broken(student_code)
    assert student_model is not None, (
        "No EfficientNet model found; cannot run hidden inference."
    )
    assert hidden_inference is not None, (
        "Hidden inference failed. Make sure the model is in eval mode and accepts "
        "image tensors of shape (1, 3, H, W)."
    )
    assert len(hidden_inference) == 90, (
        f"Expected 90 hidden-image predictions, got {len(hidden_inference)}."
    )
    invalid = [p for p in hidden_inference if p not in valid_categories]
    assert len(invalid) == 0, (
        f"Found {len(invalid)} hidden prediction(s) that are not valid ImageNet category names."
    )
    unique_classes = set(hidden_inference)
    assert len(unique_classes) >= 2, (
        f"Expected predictions to span at least 2 different classes, "
        f"got {len(unique_classes)}: {unique_classes}. "
        "Ensure the model is performing live inference, not returning a constant."
    )
