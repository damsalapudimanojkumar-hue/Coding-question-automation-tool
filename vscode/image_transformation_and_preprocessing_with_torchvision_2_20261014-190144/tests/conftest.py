import pytest
import time
import json
import os
import tempfile
import zipfile

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
# Core fixture: run the student notebook
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# eval_transform fixture (name-first, type fallback)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def eval_transform(student_code):
    """Retrieve eval_transform: try by name first, then type scan."""
    if student_code.get("__error__"):
        return None
    # Name-first lookup (problem statement explicitly names this variable)
    candidate = student_code.get("eval_transform", None)
    if isinstance(candidate, transforms.Compose):
        return candidate
    # Type fallback: find a Compose that has NO stochastic transforms
    for name, val in student_code.items():
        if name.startswith("__"):
            continue
        if not isinstance(val, transforms.Compose):
            continue
        has_stochastic = any(isinstance(t, _STOCHASTIC_TYPES) for t in val.transforms)
        has_totensor   = any(isinstance(t, transforms.ToTensor) for t in val.transforms)
        if not has_stochastic and has_totensor:
            return val
    return None


# ---------------------------------------------------------------------------
# train_transform fixture (name-first, type fallback)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def train_transform(student_code):
    """Retrieve train_transform: try by name first, then type scan."""
    if student_code.get("__error__"):
        return None
    candidate = student_code.get("train_transform", None)
    if isinstance(candidate, transforms.Compose):
        return candidate
    # Fallback: a Compose that contains at least one stochastic transform
    for name, val in student_code.items():
        if name.startswith("__"):
            continue
        if not isinstance(val, transforms.Compose):
            continue
        has_stochastic = any(isinstance(t, _STOCHASTIC_TYPES) for t in val.transforms)
        if has_stochastic:
            return val
    return None


# ---------------------------------------------------------------------------
# model fixture (nn.Module with trainable parameters)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model(student_code):
    """Detect a trained torch.nn.Module with at least one trainable parameter."""
    if student_code.get("__error__"):
        return None
    # Name-first (problem statement explicitly names this variable)
    candidate = student_code.get("model", None)
    if (
        candidate is not None
        and isinstance(candidate, nn.Module)
        and any(p.requires_grad for p in candidate.parameters())
    ):
        return candidate
    # Type scan fallback
    for name, val in student_code.items():
        if name.startswith("__"):
            continue
        if isinstance(val, nn.Module) and any(p.requires_grad for p in val.parameters()):
            return val
    return None


# ---------------------------------------------------------------------------
# hidden test accuracy fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def hidden_test_accuracy(model, eval_transform):
    """
    Extract tests/test_images_CV3.zip into a temp dir, load with ImageFolder,
    and compute accuracy using the student's model and eval_transform.
    Returns float in [0, 1] or None if either artefact is missing.
    """
    if model is None or eval_transform is None:
        return None

    test_zip = os.path.join("tests", "test_images_CV3.zip")
    if not os.path.isfile(test_zip):
        return None

    tmp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(test_zip) as z:
            z.extractall(tmp_dir)

        test_dataset = ImageFolder(
            os.path.join(tmp_dir, "test"),
            transform=eval_transform,
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                preds    = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        return correct / total if total > 0 else None
    except Exception:
        return None
