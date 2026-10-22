import csv
import json
import os
import tempfile
import time
import zipfile

import pytest
import numpy as np
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import EfficientNet
from PIL import Image


# ---------------------------------------------------------------------------
# Image dataset helpers (verbatim from platform spec)
# ---------------------------------------------------------------------------

IMAGE_EXTS = (
    '.jpg', '.jpeg', '.png', '.bmp', '.webp',
    '.ppm', '.pgm', '.tif', '.tiff', '.gif'
)


def iter_dataset(root, spec):
    """Yield (image path, label) for every image `spec` describes inside
    the extracted archive at `root`.  label is a string, or None for an
    unlabelled dataset."""

    def join(rel):
        return os.path.join(root, *[p for p in rel.split("/") if p])

    if spec["kind"] == "folders":
        for top in spec["roots"]:
            for label in sorted(os.listdir(join(top))):
                folder = os.path.join(join(top), label)
                if os.path.isdir(folder):
                    for name in sorted(os.listdir(folder)):
                        if name.lower().endswith(IMAGE_EXTS):
                            yield os.path.join(folder, name), label
    elif spec["kind"] == "csv":
        by_name = None
        for table in spec["tables"]:
            if table["base"] == "*" and by_name is None:
                by_name = {
                    f: os.path.join(d, f)
                    for d, _, files in os.walk(root)
                    for f in files
                }
            with open(join(table["path"]), newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    name = row[table["file_column"]].strip() + table["suffix"]
                    if table["base"] == "*":
                        path = by_name.get(
                            name.replace("\\", "/").split("/")[-1],
                            join(name),
                        )
                    else:
                        path = join(table["base"] + "/" + name.replace("\\", "/"))
                    yield path, row[table["label_column"]].strip()
    else:
        for top in spec["dirs"]:
            for name in sorted(os.listdir(join(top))):
                if name.lower().endswith(IMAGE_EXTS):
                    yield os.path.join(join(top), name), None


HIDDEN_LAYOUT = {"kind": "folders", "roots": ["animals/animals"]}


# ---------------------------------------------------------------------------
# Notebook loader
# ---------------------------------------------------------------------------

def _load_notebook_code(path):
    """Parse .ipynb JSON and return concatenated source of all code cells,
    with magic lines stripped."""
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def student_code():
    """Execute the student notebook and return its namespace."""
    student_vars = {"__error__": None, "__exec_time__": None}
    start = time.time()
    try:
        code = _load_notebook_code("solution.ipynb")
        exec(compile(code, "solution.ipynb", "exec"), student_vars)
    except Exception as e:
        student_vars["__error__"] = e
    student_vars["__exec_time__"] = time.time() - start
    return student_vars


@pytest.fixture(scope="session")
def student_model(student_code):
    """Detect EfficientNet model in student namespace.

    The problem statement explicitly names the variable 'model', so we try
    that name first, then fall back to a type scan.
    """
    if student_code.get("__error__"):
        return None

    # Try by explicit name first
    candidate = student_code.get("model", None)
    if candidate is not None and isinstance(candidate, EfficientNet):
        return candidate

    # Fall back to type scan
    for name, val in student_code.items():
        if name.startswith("__"):
            continue
        if isinstance(val, EfficientNet):
            return val

    return None


@pytest.fixture(scope="session")
def student_predictions(student_code):
    """Retrieve the student's 'predictions' list from the namespace."""
    if student_code.get("__error__"):
        return None

    preds = student_code.get("predictions", None)
    if preds is not None and isinstance(preds, list):
        return preds
    return None


@pytest.fixture(scope="session")
def valid_categories():
    """Return the full set of ImageNet category names for DEFAULT weights."""
    weights = EfficientNet_B0_Weights.DEFAULT
    return set(weights.meta["categories"])


@pytest.fixture(scope="session")
def hidden_inference(student_model):
    """Run student model on all 90 hidden test images.

    Extracts tests/test_images_TRNSF1.zip into a temp directory, iterates
    via iter_dataset, and returns the list of predicted class-name strings.
    Returns None on any failure.
    """
    if student_model is None:
        return None

    try:
        weights = EfficientNet_B0_Weights.DEFAULT
        transform = weights.transforms()
        categories = weights.meta["categories"]

        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile("tests/test_images_TRNSF1.zip") as zf:
            zf.extractall(tmp)

        hidden_preds = []
        for img_path, _label in iter_dataset(tmp, HIDDEN_LAYOUT):
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = student_model(tensor)
            pred_idx = output.argmax(dim=1).item()
            hidden_preds.append(categories[pred_idx])

        return hidden_preds
    except Exception:
        return None
