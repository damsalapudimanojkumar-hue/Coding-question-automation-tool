import pytest
import time
import json
import os
import random
import tempfile
import zipfile

import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms import (
    Compose, Resize, ToTensor, Normalize, RandomErasing,
    RandomHorizontalFlip, RandomVerticalFlip, RandomCrop,
    RandomResizedCrop, RandomRotation, RandomAffine,
    ColorJitter, RandomGrayscale, GaussianBlur,
)
from PIL import Image


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


def flatten_compose(ts):
    """Recursively flatten nested Compose objects into a single list."""
    result = []
    for t in ts:
        if isinstance(t, Compose):
            result.extend(flatten_compose(t.transforms))
        else:
            result.append(t)
    return result


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def student_code():
    """Execute the student notebook and return its global namespace."""
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
def train_transform(student_code):
    """Look up train_transform by exact name."""
    if student_code.get("__error__"):
        return None
    val = student_code.get("train_transform", None)
    if isinstance(val, Compose):
        return val
    return None


@pytest.fixture(scope="session")
def test_transform(student_code):
    """Look up test_transform by exact name."""
    if student_code.get("__error__"):
        return None
    val = student_code.get("test_transform", None)
    if isinstance(val, Compose):
        return val
    return None


@pytest.fixture(scope="session")
def train_data(student_code):
    """Look up train_data by exact name."""
    if student_code.get("__error__"):
        return None
    val = student_code.get("train_data", None)
    if isinstance(val, torchvision.datasets.ImageFolder):
        return val
    return None


@pytest.fixture(scope="session")
def test_data(student_code):
    """Look up test_data by exact name."""
    if student_code.get("__error__"):
        return None
    val = student_code.get("test_data", None)
    if isinstance(val, torchvision.datasets.ImageFolder):
        return val
    return None


@pytest.fixture(scope="session")
def sample_images():
    """Three synthetic PIL images at different sizes for transform checks."""
    np.random.seed(0)
    return [
        Image.fromarray(np.random.randint(0, 255, (80, 300, 3), dtype=np.uint8)),   # wide
        Image.fromarray(np.random.randint(0, 255, (300, 80, 3), dtype=np.uint8)),   # tall
        Image.fromarray(np.random.randint(0, 255, (700, 900, 3), dtype=np.uint8)),  # large
    ]
