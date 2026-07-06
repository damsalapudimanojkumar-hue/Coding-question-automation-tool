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

## Detection Patterns by Object Type

Detection patterns (classifier, scaler, probability array, PyTorch model,
clustering, name-first fallback, etc.) are maintained in one place:
`knowledge/skills/detection_patterns.md`. Use those; do not duplicate them here.