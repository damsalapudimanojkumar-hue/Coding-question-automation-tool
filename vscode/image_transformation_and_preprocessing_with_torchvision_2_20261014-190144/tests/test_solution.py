import pytest
import torch
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _skip_if_broken(student_code):
    if student_code.get("__error__"):
        pytest.skip("solution.ipynb raised an exception -- skipping dependent test")


_STOCHASTIC_TYPES = (
    transforms.RandomHorizontalFlip,
    transforms.RandomVerticalFlip,
    transforms.ColorJitter,
    transforms.RandomRotation,
    transforms.RandomCrop,
    transforms.RandomAffine,
    transforms.RandomGrayscale,
    transforms.RandomPerspective,
)


# ---------------------------------------------------------------------------
# PY1 -- Notebook executes without error
# ---------------------------------------------------------------------------

def test_code_runs_without_error_PY1(student_code):
    """_TC1: solution.ipynb must exec() cleanly with no exceptions."""
    err = student_code.get("__error__")
    assert err is None, f"{type(err).__name__}: {err}"


# ---------------------------------------------------------------------------
# PY2 -- eval_transform has the required deterministic structure
# ---------------------------------------------------------------------------

def test_eval_transform_structure_PY2(student_code, eval_transform):
    """_TC2: eval_transform must be a Compose with ToTensor before Normalize, no stochastic ops."""
    _skip_if_broken(student_code)

    assert eval_transform is not None, (
        "eval_transform not found or is not a transforms.Compose"
    )

    ts = eval_transform.transforms

    # (a) ToTensor present
    totensor_indices = [i for i, t in enumerate(ts) if isinstance(t, transforms.ToTensor)]
    assert len(totensor_indices) > 0, "ToTensor not found in eval_transform"

    # (b) Normalize present
    normalize_indices = [i for i, t in enumerate(ts) if isinstance(t, transforms.Normalize)]
    assert len(normalize_indices) > 0, "Normalize not found in eval_transform"

    # (c) ToTensor appears before Normalize
    assert totensor_indices[0] < normalize_indices[0], (
        "ToTensor must appear before Normalize in eval_transform"
    )

    # (d) No stochastic transforms present
    stochastic_found = [type(t).__name__ for t in ts if isinstance(t, _STOCHASTIC_TYPES)]
    assert len(stochastic_found) == 0, (
        f"eval_transform must not contain stochastic transforms; found: {stochastic_found}"
    )


# ---------------------------------------------------------------------------
# PY3 -- train_transform has all required augmentations and correct stats
# ---------------------------------------------------------------------------

def test_train_transform_structure_PY3(student_code, train_transform):
    """_TC3: train_transform must contain required augmentations and EuroSAT normalisation."""
    _skip_if_broken(student_code)

    assert train_transform is not None, (
        "train_transform not found or is not a transforms.Compose"
    )

    ts = train_transform.transforms

    # (a) RandomHorizontalFlip present
    assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in ts), (
        "RandomHorizontalFlip not found in train_transform"
    )

    # (b) RandomVerticalFlip present
    assert any(isinstance(t, transforms.RandomVerticalFlip) for t in ts), (
        "RandomVerticalFlip not found in train_transform"
    )

    # (c) ColorJitter present
    assert any(isinstance(t, transforms.ColorJitter) for t in ts), (
        "ColorJitter not found in train_transform"
    )

    # (d) Normalize present with correct EuroSAT stats (tolerance 0.001)
    norms = [t for t in ts if isinstance(t, transforms.Normalize)]
    assert len(norms) > 0, "Normalize not found in train_transform"

    norm = norms[0]
    expected_mean = [0.3444, 0.3803, 0.4078]
    expected_std  = [0.2038, 0.1367, 0.1148]

    actual_mean = list(norm.mean)
    actual_std  = list(norm.std)

    for i, (a, e) in enumerate(zip(actual_mean, expected_mean)):
        assert abs(a - e) < 0.001, (
            f"Normalize mean channel {i}: expected {e}, got {a:.4f}"
        )
    for i, (a, e) in enumerate(zip(actual_std, expected_std)):
        assert abs(a - e) < 0.001, (
            f"Normalize std channel {i}: expected {e}, got {a:.4f}"
        )

    # (e) ToTensor appears before Normalize
    totensor_indices  = [i for i, t in enumerate(ts) if isinstance(t, transforms.ToTensor)]
    normalize_indices = [i for i, t in enumerate(ts) if isinstance(t, transforms.Normalize)]
    assert len(totensor_indices) > 0, "ToTensor not found in train_transform"
    assert totensor_indices[0] < normalize_indices[0], (
        "ToTensor must appear before Normalize in train_transform"
    )


# ---------------------------------------------------------------------------
# PY4 -- Trained model achieves >= 0.65 accuracy on hidden test images
# ---------------------------------------------------------------------------

def test_model_accuracy_on_hidden_test_PY4(student_code, model, hidden_test_accuracy):
    """_TC4: model must achieve >= 0.65 accuracy on the hidden EuroSAT test patches."""
    _skip_if_broken(student_code)

    assert model is not None, (
        "No trained torch.nn.Module found in solution"
    )

    assert hidden_test_accuracy is not None, (
        "Could not compute test accuracy -- check that model and eval_transform are defined"
    )

    assert hidden_test_accuracy >= 0.65, (
        f"Test accuracy = {hidden_test_accuracy:.2f}, required >= 0.65"
    )
