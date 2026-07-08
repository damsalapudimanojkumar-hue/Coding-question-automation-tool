import pytest
import time
import json
import numpy as np


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


def _callables(student_code):
    return [v for k, v in student_code.items()
            if not k.startswith("__") and callable(v)]


def _safe_call(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def _is_2d_array(x):
    return isinstance(x, np.ndarray) and x.ndim == 2


# ---- Function detectors: find by BEHAVIOR, not by name ----

@pytest.fixture(scope="session")
def box_fn(student_code):
    """A 1-arg function returning a uniform NxN array summing to ~1."""
    if student_code.get("__error__"):
        return None
    for fn in _callables(student_code):
        out = _safe_call(fn, 3)
        if _is_2d_array(out) and out.shape == (3, 3):
            if abs(out.sum() - 1.0) < 1e-6 and np.allclose(out, out.flat[0]):
                return fn
    return None


@pytest.fixture(scope="session")
def gaussian_fn(student_code):
    """A 2-arg function returning a normalized symmetric NxN array, center is the max."""
    if student_code.get("__error__"):
        return None
    for fn in _callables(student_code):
        out = _safe_call(fn, 5, 1.0)
        if _is_2d_array(out) and out.shape == (5, 5):
            if abs(out.sum() - 1.0) < 1e-6 and out[2, 2] == out.max() and not np.allclose(out, out.flat[0]):
                return fn
    return None


@pytest.fixture(scope="session")
def sharpen_fn(student_code):
    """A 0-arg function returning the 3x3 sharpen kernel summing to 1 with center 5."""
    if student_code.get("__error__"):
        return None
    for fn in _callables(student_code):
        out = _safe_call(fn)
        if _is_2d_array(out) and out.shape == (3, 3):
            if abs(out.sum() - 1.0) < 1e-6 and abs(out[1, 1] - 5) < 1e-6:
                return fn
    return None


@pytest.fixture(scope="session")
def sobel_fn(student_code):
    """A 0-arg function returning a tuple of two 3x3 arrays, each summing to 0."""
    if student_code.get("__error__"):
        return None
    for fn in _callables(student_code):
        out = _safe_call(fn)
        if isinstance(out, tuple) and len(out) == 2:
            a, b = out
            if _is_2d_array(a) and _is_2d_array(b) and a.shape == (3, 3) and b.shape == (3, 3):
                if abs(a.sum()) < 1e-6 and abs(b.sum()) < 1e-6:
                    return fn
    return None


@pytest.fixture(scope="session")
def apply_filter_fn(student_code):
    """
    A 2-arg (image, kernel) function performing cross-CORRELATION, zero-padded,
    same size. Identity kernel returns the image unchanged AND an asymmetric
    shift kernel shifts in the correlation direction (distinguishes it from
    convolve, which flips the kernel).
    """
    if student_code.get("__error__"):
        return None
    img = np.arange(25, dtype=float).reshape(5, 5)
    identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    probe = np.zeros((5, 5), dtype=float)
    probe[2, 2] = 100.0
    shift = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    for fn in _callables(student_code):
        out = _safe_call(fn, img, identity)
        if not (_is_2d_array(out) and out.shape == img.shape and np.allclose(out, img, atol=1e-6)):
            continue
        shifted = _safe_call(fn, probe, shift)
        if not (_is_2d_array(shifted) and shifted.shape == probe.shape):
            continue
        # correlation: source 100 lands at [2,1] (kernel slid right)
        if abs(shifted[2, 1] - 100.0) < 1e-6 and abs(shifted[2, 3]) < 1e-6:
            return fn
    return None


@pytest.fixture(scope="session")
def convolve_fn(student_code, apply_filter_fn):
    """
    A 2-arg (image, kernel) function performing TRUE convolution.
    With the same asymmetric shift kernel, convolution must shift opposite
    to correlation (source lands at [2,3]).
    """
    if student_code.get("__error__"):
        return None
    img = np.zeros((5, 5), dtype=float)
    img[2, 2] = 100.0
    shift = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    for fn in _callables(student_code):
        out = _safe_call(fn, img, shift)
        if _is_2d_array(out) and out.shape == img.shape:
            if abs(out[2, 3] - 100.0) < 1e-6 and abs(out[2, 1]) < 1e-6:
                return fn
    return None


@pytest.fixture(scope="session")
def images_dir():
    return "images"
