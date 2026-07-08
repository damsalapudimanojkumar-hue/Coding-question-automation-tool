# conftest.py — Reference Format

## Mandatory Rules
- All fixtures: `scope="session"` — no exceptions
- Never `pytest.fail()` inside fixtures — store error in `__exec_error__`, let test functions assert
- Always `encoding="utf-8"` in `open()`
- Strip `%`, `!`, `?` lines before `exec()` using the magic-line stripper
- Detect objects by `isinstance` and structure — never by variable name
  - Exception: if problem statement explicitly names a variable, try that name first, then fall back to type scan
- No hooks (no `pytest_runtest_makereport`, no `pytest_sessionfinish`)
- Ground truth loads from S3 URL — never from local file path in the platform version

## Canonical Structure
```python
import pytest
import nbformat
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# ... other imports matching the assignment topic

def _strip_magic_lines(source: str) -> str:
    cleaned = []
    for line in source.splitlines():
        if line.strip().startswith(("%", "!", "?")):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

@pytest.fixture(scope="session")
def student_code():
    notebook_path = "solution.ipynb"
    error = None
    student_ns = {}
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        combined_source = ""
        for cell in nb.cells:
            if cell.cell_type == "code":
                combined_source += _strip_magic_lines(cell.source) + "\n"
        exec(compile(combined_source, notebook_path, "exec"), student_ns)
    except Exception as e:
        error = e
    student_ns["__exec_error__"] = error
    return student_ns

@pytest.fixture(scope="session")
def ground_truth():
    # Platform version: load from S3
    return pd.read_csv("https://s3.ap-south-1.amazonaws.com/new-assets.ccbp.in/frontend/content/aiml/<path>/<filename>.csv")
    # Local testing version: load from tests/ground_truth.csv
```

## Image / Computer-Vision variant

For CV assignments the dataset is real image files, not CSVs. The student loads
`dataset_train_<CODE>/` with `torchvision.datasets.ImageFolder`; the grader scores
the flat, unlabeled `dataset_test_<CODE>/` images against
`tests/ground_truth_<CODE>.csv` (`filename,label`).

```python
import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

@pytest.fixture(scope="session")
def ground_truth():
    # Platform version: load the manifest from S3; local version: from tests/
    gt = pd.read_csv("tests/ground_truth_<CODE>.csv")   # columns: filename,label
    tfm = transforms.Compose([
        transforms.Resize((32, 32)),   # SAME preprocessing the problem specifies
        transforms.ToTensor(),
    ])
    test_dir = "dataset_test_<CODE>"
    images, labels = [], []
    for _, row in gt.iterrows():
        img = Image.open(os.path.join(test_dir, row["filename"])).convert("RGB")
        images.append(tfm(img))
        labels.append(row["label"])
    return torch.stack(images), labels   # align predictions to labels by row order
```

Detect the student model by TYPE + BEHAVIOR (isinstance `nn.Module` plus a
forward-pass shape probe), never by variable name. Compute accuracy by running the
detected model over these test images and comparing argmax predictions to the labels.

## Detection Patterns by Object Type

Detection patterns (classifier, scaler, probability array, PyTorch model,
CV model behavioral probe / architecture inspection, clustering, name-first
fallback, etc.) are maintained in one place:
`knowledge/skills/detection_patterns.md`. Use those; do not duplicate them here.