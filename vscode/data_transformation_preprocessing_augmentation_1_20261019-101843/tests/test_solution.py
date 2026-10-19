"""
Test suite for the Recycling Bin Classifier: Smart Waste Sorting with Image Transforms.
"""
import pytest
import random
import numpy as np
import torch
import torchvision
from torchvision.transforms import (
    Compose, Resize, ToTensor, Normalize, RandomErasing,
    RandomHorizontalFlip, RandomVerticalFlip, RandomCrop,
    RandomResizedCrop, RandomRotation, RandomAffine,
    ColorJitter, RandomGrayscale, GaussianBlur,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_broken(student_code):
    if student_code.get("__error__"):
        pytest.skip("solution.ipynb raised an exception during execution.")


def flatten_compose(ts):
    """Recursively flatten nested Compose objects into a flat list."""
    result = []
    for t in ts:
        if isinstance(t, Compose):
            result.extend(flatten_compose(t.transforms))
        else:
            result.append(t)
    return result


# ---------------------------------------------------------------------------
# PY1 — Code runs without error
# ---------------------------------------------------------------------------

def test_code_runs_without_error_PY1(student_code):
    """PY1: The notebook must execute end-to-end without raising any exception."""
    err = student_code.get("__error__")
    assert err is None, f"Notebook raised {type(err).__name__}: {err}"


# ---------------------------------------------------------------------------
# PY2 — test_transform is exactly Resize -> ToTensor -> Normalize, deterministic
# ---------------------------------------------------------------------------

def test_test_transform_structure_and_determinism_PY2(
    student_code, test_transform, train_transform, sample_images
):
    """PY2: test_transform must be Resize(64,64) -> ToTensor -> Normalize only,
    use the same Normalize stats as train_transform, and produce identical
    [3,64,64] float32 tensors on every call."""
    _skip_if_broken(student_code)

    # --- test_transform must exist ---
    assert test_transform is not None, (
        "test_transform not found. Make sure you define it as a transforms.Compose object."
    )
    assert train_transform is not None, (
        "train_transform not found. Make sure you define it as a transforms.Compose object."
    )

    flat_test = flatten_compose(test_transform.transforms)

    # --- Allowlist: exactly 3 transforms in the correct order ---
    assert len(flat_test) == 3, (
        f"test_transform should have exactly 3 transforms (Resize, ToTensor, Normalize), "
        f"found {len(flat_test)}: {[type(t).__name__ for t in flat_test]}"
    )
    assert isinstance(flat_test[0], Resize), (
        "First transform in test_transform must be Resize."
    )
    assert isinstance(flat_test[1], ToTensor), (
        "Second transform in test_transform must be ToTensor."
    )
    assert isinstance(flat_test[2], Normalize), (
        "Third transform in test_transform must be Normalize."
    )

    # --- Resize must use (64, 64) ---
    resize_t = flat_test[0]
    assert list(resize_t.size) == [64, 64], (
        f"Resize in test_transform must target (64, 64), got size={resize_t.size}. "
        "Use Resize((64, 64)) not Resize(64)."
    )

    # --- Normalize mean/std must match train_transform ---
    flat_train = flatten_compose(train_transform.transforms)
    assert any(isinstance(t, Normalize) for t in flat_train), (
        "train_transform must contain a Normalize transform."
    )
    assert any(isinstance(t, Normalize) for t in flat_test), (
        "test_transform must contain a Normalize transform."
    )
    n_train = next(t for t in flat_train if isinstance(t, Normalize))
    n_test  = next(t for t in flat_test  if isinstance(t, Normalize))
    assert list(n_train.mean) == list(n_test.mean), (
        "test_transform Normalize mean must match train_transform Normalize mean."
    )
    assert list(n_train.std) == list(n_test.std), (
        "test_transform Normalize std must match train_transform Normalize std."
    )

    # --- Output shape and dtype for all three sample images ---
    for i, img in enumerate(sample_images):
        out = test_transform(img)
        assert out.shape == torch.Size([3, 64, 64]), (
            f"test_transform output shape for sample image {i} is {out.shape}, "
            "expected torch.Size([3, 64, 64])."
        )
        assert out.dtype == torch.float32, (
            f"test_transform output dtype for sample image {i} is {out.dtype}, "
            "expected torch.float32."
        )

    # --- Determinism: same output on two consecutive calls ---
    for i, img in enumerate(sample_images):
        out1 = test_transform(img)
        out2 = test_transform(img)
        assert torch.equal(out1, out2), (
            f"test_transform is not deterministic for sample image {i}. "
            "Remove all augmentation from test_transform."
        )


# ---------------------------------------------------------------------------
# PY3 — train_transform includes geometric, colour, and RandomErasing
# ---------------------------------------------------------------------------

def test_train_transform_augmentations_PY3(student_code, train_transform):
    """PY3: train_transform must contain at least one geometric augmentation,
    at least one colour augmentation, and RandomErasing."""
    _skip_if_broken(student_code)

    assert train_transform is not None, (
        "train_transform not found. Make sure you define it as a transforms.Compose object."
    )

    flat_train = flatten_compose(train_transform.transforms)

    geo_types = (
        RandomHorizontalFlip, RandomVerticalFlip, RandomCrop,
        RandomResizedCrop, RandomRotation, RandomAffine,
    )
    col_types = (ColorJitter, RandomGrayscale, GaussianBlur)

    has_geo   = any(isinstance(t, geo_types) for t in flat_train)
    has_col   = any(isinstance(t, col_types) for t in flat_train)
    has_erase = any(isinstance(t, RandomErasing) for t in flat_train)

    assert has_geo, (
        "train_transform must include at least one geometric augmentation "
        "(e.g. RandomHorizontalFlip, RandomRotation, RandomCrop, RandomResizedCrop)."
    )
    assert has_col, (
        "train_transform must include at least one colour/appearance augmentation "
        "(e.g. ColorJitter, RandomGrayscale, GaussianBlur)."
    )
    assert has_erase, (
        "train_transform must include RandomErasing (applied after ToTensor)."
    )


# ---------------------------------------------------------------------------
# PY4 — Transform ordering is correct in train_transform
# ---------------------------------------------------------------------------

def test_train_transform_ordering_PY4(student_code, train_transform, sample_images):
    """PY4: All PIL-level transforms must appear before ToTensor; Normalize and
    RandomErasing must appear after ToTensor. Pipeline must not raise on all
    three sample images."""
    _skip_if_broken(student_code)

    assert train_transform is not None, (
        "train_transform not found."
    )

    flat_train = flatten_compose(train_transform.transforms)

    # ToTensor and Normalize must exist before we do index lookups
    assert any(isinstance(t, ToTensor) for t in flat_train), (
        "train_transform must contain ToTensor."
    )
    assert any(isinstance(t, Normalize) for t in flat_train), (
        "train_transform must contain Normalize."
    )
    assert any(isinstance(t, RandomErasing) for t in flat_train), (
        "train_transform must contain RandomErasing."
    )

    tt_idx = next(i for i, t in enumerate(flat_train) if isinstance(t, ToTensor))
    n_idx  = next(i for i, t in enumerate(flat_train) if isinstance(t, Normalize))
    re_idx = next(i for i, t in enumerate(flat_train) if isinstance(t, RandomErasing))

    # All image-level transforms must come before ToTensor
    img_level_types = (
        Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomCrop,
        RandomResizedCrop, RandomRotation, RandomAffine,
        ColorJitter, RandomGrayscale, GaussianBlur,
    )
    for i, t in enumerate(flat_train):
        if isinstance(t, img_level_types):
            assert i < tt_idx, (
                f"{type(t).__name__} at index {i} must appear before ToTensor "
                f"(index {tt_idx}). PIL-level transforms must precede ToTensor."
            )

    assert n_idx > tt_idx, (
        f"Normalize (index {n_idx}) must come after ToTensor (index {tt_idx})."
    )
    assert re_idx > tt_idx, (
        f"RandomErasing (index {re_idx}) must come after ToTensor (index {tt_idx})."
    )

    # Pipeline must run without error on all sample images
    for i, img in enumerate(sample_images):
        try:
            out = train_transform(img)
        except Exception as e:
            pytest.fail(
                f"train_transform raised an exception on sample image {i}: "
                f"{type(e).__name__}: {e}"
            )


# ---------------------------------------------------------------------------
# PY5 — train_transform output is [3, 64, 64] float32 and stochastic
# ---------------------------------------------------------------------------

def test_train_transform_output_and_stochasticity_PY5(
    student_code, train_transform, sample_images
):
    """PY5: train_transform must produce [3,64,64] float32 tensors from all sample
    images, and must be stochastic (at least one pair of outputs must differ when
    applied 10 times to the same image)."""
    _skip_if_broken(student_code)

    assert train_transform is not None, (
        "train_transform not found."
    )

    # Shape and dtype checks
    for i, img in enumerate(sample_images):
        out = train_transform(img)
        assert out.shape == torch.Size([3, 64, 64]), (
            f"train_transform output shape for sample image {i} is {out.shape}, "
            "expected torch.Size([3, 64, 64])."
        )
        assert out.dtype == torch.float32, (
            f"train_transform output dtype for sample image {i} is {out.dtype}, "
            "expected torch.float32."
        )

    # Stochasticity: 10 applications of train_transform must not all be identical
    reference_img = sample_images[0]
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    outputs = [train_transform(reference_img) for _ in range(10)]
    all_identical = all(torch.equal(outputs[0], o) for o in outputs[1:])

    assert not all_identical, (
        "train_transform produced identical output on all 10 applications to the "
        "same image. Ensure your training pipeline contains stochastic augmentations "
        "with non-zero probability."
    )


# ---------------------------------------------------------------------------
# PY6 — train_data and test_data are correctly configured ImageFolder objects
# ---------------------------------------------------------------------------

def test_datasets_configured_correctly_PY6(
    student_code, train_data, test_data, train_transform, test_transform
):
    """PY6: train_data and test_data must be ImageFolder objects with the correct
    transforms attached (not swapped, not the same object), each with 360 images."""
    _skip_if_broken(student_code)

    # Both datasets must exist
    assert train_data is not None, (
        "train_data not found. Make sure you create an ImageFolder dataset with "
        "transform=train_transform and store it in the variable 'train_data'."
    )
    assert test_data is not None, (
        "test_data not found. Make sure you create an ImageFolder dataset with "
        "transform=test_transform and store it in the variable 'test_data'."
    )

    # Type checks
    assert isinstance(train_data, torchvision.datasets.ImageFolder), (
        "train_data must be a torchvision.datasets.ImageFolder instance."
    )
    assert isinstance(test_data, torchvision.datasets.ImageFolder), (
        "test_data must be a torchvision.datasets.ImageFolder instance."
    )

    # transform must exist to compare
    assert train_transform is not None, "train_transform not found."
    assert test_transform is not None, "test_transform not found."

    # Correct transforms attached (identity check)
    assert train_data.transform is train_transform, (
        "train_data.transform must be train_transform (not test_transform or a copy)."
    )
    assert test_data.transform is test_transform, (
        "test_data.transform must be test_transform (not train_transform or a copy)."
    )

    # Pipelines must not be the same object on both datasets
    assert train_data.transform is not test_data.transform, (
        "train_data and test_data must use different transform objects. "
        "Do not assign the same pipeline to both datasets."
    )

    # Swapped-pipeline guards
    assert train_data.transform is not test_transform, (
        "train_data appears to have test_transform attached. Attach train_transform to train_data."
    )
    assert test_data.transform is not train_transform, (
        "test_data appears to have train_transform attached. Attach test_transform to test_data."
    )

    # Length checks: student archive has 360 images (72 per class x 5 classes)
    assert len(train_data) == 360, (
        f"train_data should contain 360 images (72 per class x 5 classes), "
        f"found {len(train_data)}. Make sure you point ImageFolder at images/garbage_classification/."
    )
    assert len(test_data) == 360, (
        f"test_data should contain 360 images (72 per class x 5 classes), "
        f"found {len(test_data)}. Make sure you point ImageFolder at images/garbage_classification/."
    )
