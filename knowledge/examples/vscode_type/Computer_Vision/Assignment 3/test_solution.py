import pytest
import os
import numpy as np
from PIL import Image


# Small sample keeps grading fast regardless of dataset size.
# Tiles span the full 900-image range so detection of category-induced edge cases is broad.
SAMPLE_TILES = ["tile_001.jpg", "tile_100.jpg", "tile_250.jpg",
                "tile_400.jpg", "tile_500.jpg"]


def _load(images_dir, fname):
    return np.array(Image.open(os.path.join(images_dir, fname)).convert("L"))


def _ref_apply(image, kernel):
    """Reference cross-correlation, zero padded, same size."""
    img = image.astype(float)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(img, dtype=float)
    H, W = img.shape
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
    return out


def _skip_if_broken(student_code):
    if student_code.get("__error__"):
        pytest.skip("solution.ipynb failed to execute")


# -------------------- PY1 --------------------
def test_code_runs_without_error_PY1(student_code):
    """_TC1"""
    err = student_code.get("__error__")
    assert err is None, f"{type(err).__name__}: {err}"


# -------------------- PY2 --------------------
def test_box_kernel_PY2(student_code, box_fn):
    """_TC2"""
    _skip_if_broken(student_code)
    assert box_fn is not None, "No box_kernel function found (size x size, all equal, sums to 1)."
    for n in (3, 5):
        k = box_fn(n)
        assert k.shape == (n, n), f"box_kernel({n}) shape {k.shape}, expected ({n}, {n})."
        assert abs(k.sum() - 1.0) < 1e-6, f"box_kernel({n}) sums to {k.sum():.4f}, expected 1."
        assert np.allclose(k, 1.0 / (n * n)), "box_kernel entries are not all equal to 1/(n*n)."


# -------------------- PY3 --------------------
def test_gaussian_kernel_PY3(student_code, gaussian_fn):
    """_TC3"""
    _skip_if_broken(student_code)
    assert gaussian_fn is not None, "No gaussian_kernel function found (normalized, symmetric, center peak)."
    k = gaussian_fn(5, 1.0)
    assert k.shape == (5, 5), f"gaussian_kernel(5,1.0) shape {k.shape}, expected (5,5)."
    assert abs(k.sum() - 1.0) < 1e-6, f"gaussian sums to {k.sum():.4f}, expected 1."
    assert k[2, 2] == k.max(), "Gaussian peak is not at the center."
    assert np.allclose(k, k.T, atol=1e-8), "Gaussian kernel is not symmetric."
    assert np.allclose(k, np.flipud(k), atol=1e-8), "Gaussian kernel is not vertically symmetric."


# -------------------- PY4 --------------------
def test_sharpen_kernel_PY4(student_code, sharpen_fn):
    """_TC4"""
    _skip_if_broken(student_code)
    assert sharpen_fn is not None, "No sharpen_kernel function found (3x3, center 5, sums to 1)."
    k = sharpen_fn()
    expected = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)
    assert k.shape == (3, 3), f"sharpen shape {k.shape}, expected (3,3)."
    assert np.allclose(k, expected), "sharpen_kernel values do not match [[0,-1,0],[-1,5,-1],[0,-1,0]]."


# -------------------- PY5 --------------------
def test_sobel_kernels_PY5(student_code, sobel_fn):
    """_TC5"""
    _skip_if_broken(student_code)
    assert sobel_fn is not None, "No sobel_kernels function found (returns two 3x3 arrays summing to 0)."
    sx, sy = sobel_fn()
    exp_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    exp_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    match_xy = np.allclose(sx, exp_x) and np.allclose(sy, exp_y)
    match_yx = np.allclose(sx, exp_y) and np.allclose(sy, exp_x)
    assert match_xy or match_yx, "Sobel kernels do not match the expected horizontal/vertical pair."


# -------------------- PY6 --------------------
def test_apply_filter_shape_and_identity_PY6(student_code, apply_filter_fn, images_dir):
    """_TC6"""
    _skip_if_broken(student_code)
    assert apply_filter_fn is not None, "No apply_filter function found (correlation, zero-padded, same size, identity = original)."
    img = _load(images_dir, SAMPLE_TILES[0])
    identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    out = apply_filter_fn(img, identity)
    assert out.shape == img.shape, f"apply_filter output {out.shape}, expected {img.shape} (same size)."
    assert np.allclose(out, img.astype(float), atol=1e-6), "Identity kernel did not return the original image (check zero padding / same-size output)."


# -------------------- PY7 --------------------
def test_apply_filter_box_PY7(student_code, apply_filter_fn, box_fn, images_dir):
    """_TC7"""
    _skip_if_broken(student_code)
    if apply_filter_fn is None or box_fn is None:
        pytest.skip("apply_filter or box_kernel not found")
    k = box_fn(3)
    mism = []
    for fname in SAMPLE_TILES:
        img = _load(images_dir, fname)
        got = apply_filter_fn(img, k)
        exp = _ref_apply(img, k)
        if not np.allclose(got, exp, atol=1e-4):
            mism.append(f"{fname}: max diff {np.abs(got - exp).max():.4f}")
    assert not mism, f"Box-filter output mismatch: {mism[:3]}"


# -------------------- PY8 --------------------
def test_apply_filter_sobel_PY8(student_code, apply_filter_fn, sobel_fn, images_dir):
    """_TC8"""
    _skip_if_broken(student_code)
    if apply_filter_fn is None or sobel_fn is None:
        pytest.skip("apply_filter or sobel_kernels not found")
    sx, sy = sobel_fn()
    mism = []
    for fname in SAMPLE_TILES:
        img = _load(images_dir, fname)
        for k, label in ((sx, "sobel_x"), (sy, "sobel_y")):
            got = apply_filter_fn(img, k)
            exp = _ref_apply(img, k)
            if not np.allclose(got, exp, atol=1e-4):
                mism.append(f"{fname}/{label}: max diff {np.abs(got - exp).max():.4f}")
    assert not mism, f"Sobel-filter output mismatch: {mism[:3]}"


# -------------------- PY9 --------------------
def test_convolution_vs_correlation_PY9(student_code, convolve_fn, apply_filter_fn, sobel_fn, images_dir):
    """_TC9"""
    _skip_if_broken(student_code)
    assert convolve_fn is not None, "No convolve function found (true convolution = correlation with a flipped kernel)."
    if apply_filter_fn is None or sobel_fn is None:
        pytest.skip("apply_filter or sobel_kernels not found")
    sx, _ = sobel_fn()
    img = _load(images_dir, SAMPLE_TILES[0])
    conv = convolve_fn(img, sx)
    corr = apply_filter_fn(img, sx)
    assert not np.allclose(conv, corr, atol=1e-6), "convolve equals apply_filter for an asymmetric kernel (kernel was not flipped)."
    flipped = np.flipud(np.fliplr(sx))
    exp = _ref_apply(img, flipped)
    assert np.allclose(conv, exp, atol=1e-4), "convolve does not match correlation-with-flipped-kernel."


# -------------------- PY10 --------------------
def test_symmetric_kernel_equivalence_PY10(student_code, convolve_fn, apply_filter_fn, gaussian_fn, images_dir):
    """_TC10"""
    _skip_if_broken(student_code)
    if convolve_fn is None or apply_filter_fn is None or gaussian_fn is None:
        pytest.skip("convolve, apply_filter, or gaussian_kernel not found")
    g = gaussian_fn(5, 1.0)
    img = _load(images_dir, SAMPLE_TILES[1])
    conv = convolve_fn(img, g)
    corr = apply_filter_fn(img, g)
    assert np.allclose(conv, corr, atol=1e-4), "For a symmetric Gaussian, convolve and apply_filter should match."


# -------------------- PY11 --------------------
def test_execution_time_PY11(student_code):
    """_TC11"""
    t = student_code.get("__exec_time__")
    assert t is not None, "Could not measure execution time."
    assert t <= 90, f"Took {t:.1f}s, limit is 90s."
