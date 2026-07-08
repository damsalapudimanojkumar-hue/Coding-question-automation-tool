# Convolution & Filter Engine — TerraScan Aerial Imagery (Computer Vision Assignment 3)

**Topic:** Classical image filtering from scratch — kernels + cross-correlation vs true convolution (NumPy only, no deep learning)
**Dataset used:** UC Merced Land Use tiles — `images/` folder of real 256×256 color aerial tiles (question says 100 tiles / 9 land-use categories; conftest sample tiles span up to `tile_500.jpg`), converted to grayscale via a provided `load_tile` helper
**Model/algorithm:** none — six hand-written NumPy functions (`box_kernel`, `gaussian_kernel`, `sharpen_kernel`, `sobel_kernels`, `apply_filter`, `convolve`)
**Assignment type:** image / classical computer vision (algorithmic, no ML)

## Problem statement summary
Business framing: **TerraScan**, an Earth-observation pipeline, needs a filter engine to denoise/smooth aerial tiles and run edge detection before downstream classifiers. Students fill in six function stubs (correct names/signatures pre-provided) using **NumPy only** — library convolution helpers (`cv2.filter2D`, `cv2.GaussianBlur`, `cv2.Sobel`, `scipy.ndimage`/`scipy.signal`) are explicitly banned. Kernels: box (uniform, sums to 1), Gaussian (normalized, symmetric, center peak), sharpen (`[[0,-1,0],[-1,5,-1],[0,-1,0]]`), Sobel pair (each sums to 0). Filtering: `apply_filter` = zero-padded, same-size **cross-correlation**; `convolve` = true convolution = correlation with the kernel flipped on both axes. Outputs must stay floating point (no clip/`uint8`).

This is the **odd one out** of the three CV assignments — no PyTorch, no training, no accuracy metric. It's an algorithmic-correctness assignment closer in spirit to a "implement the primitive from scratch" exercise, and the hardest part is the correlation-vs-convolution distinction.

## Test cases used (PY1–PY11)
- **PY1 (5):** Code runs without error.
- **PY2 (10):** `box_kernel(size)` — size×size, all entries equal, sums to 1 (checked at n=3 and n=5).
- **PY3 (12):** `gaussian_kernel(size, sigma)` — sums to 1, peak at center, symmetric (both `k == k.T` and vertical flip).
- **PY4 (8):** `sharpen_kernel()` — exact match to `[[0,-1,0],[-1,5,-1],[0,-1,0]]`.
- **PY5 (10):** `sobel_kernels()` — returns `(sobel_x, sobel_y)`, each summing to 0; accepts either order (x,y or y,x).
- **PY6 (15):** `apply_filter` — same-size output and identity kernel returns the original image. Highest weightage.
- **PY7 (10):** `apply_filter` with a box kernel matches a reference cross-correlation implementation across sample tiles.
- **PY8 (10):** `apply_filter` with Sobel kernels matches reference edge-detection output.
- **PY9 (10):** `convolve` performs true convolution — must DIFFER from `apply_filter` on an asymmetric (Sobel) kernel, and equal correlation-with-flipped-kernel.
- **PY10 (5):** For a symmetric Gaussian kernel, `convolve` and `apply_filter` produce the same result.
- **PY11 (5):** Runs within 90 seconds.

## Threshold / calibration notes
- **No metric threshold — grading is exact numerical correctness** against a reference implementation (`_ref_apply`) with `atol=1e-4`..`1e-6`. This is fundamentally different from the ML assignments' "beat a threshold" model. The reference `_ref_apply` is the source of truth for what "correct" means (zero-padded, same-size, cross-correlation via `padded[i:i+kh, j:j+kw] * kernel`).
- **90s time limit** (vs 600s for the CNN training assignments) — pure NumPy on small grayscale tiles is fast, but a naive triple-nested Python loop over a 256×256 image × several kernels × 5 tiles can get slow, so 90s is a real (not trivial) budget that rewards vectorized or at least tight loops.
- Weightages front-load the two conceptually hard checks: PY6 (correct padding/same-size machinery, 15) and the correlation-vs-convolution pair PY9+PY10 (15 combined). The kernel-definition tests (PY2–PY5) are the easy points.

## Gotchas / things that tripped things up or are easy to get wrong
- **All six functions are detected by BEHAVIOR, not by name** (this conftest is the strongest example of the pattern in the whole set). E.g. `apply_filter` is found by: applies identity kernel = returns image unchanged, AND an asymmetric shift kernel shifts the source pixel in the *correlation* direction (`shifted[2,1]==100`); `convolve` is the function where the same shift kernel moves the pixel the *opposite* way (`out[2,3]==100`). This means a student can name the functions anything and still pass — but a student who implements `apply_filter` as true convolution (kernel flipped) will be misdetected/fail the directional probe. **The correlation-vs-convolution direction is the crux of the whole assignment** and the detector enforces it precisely.
- **`gaussian_fn` probe requires `out[2,2] == out.max()` AND `not np.allclose(out, out.flat[0])`** — i.e. it must be a genuine non-uniform Gaussian peaked at center. A student who returns a box-like or improperly normalized kernel won't be detected as the Gaussian function.
- **`convolve` must genuinely flip the kernel**, not just call `apply_filter`. PY9 asserts `convolve != apply_filter` for an asymmetric kernel — a student who defines `convolve = apply_filter` (thinking they're the same) fails PY9 but *passes* PY10 (symmetric kernel), which is a useful diagnostic signal that they missed the flip.
- **Zero padding is mandated and load-bearing** — the identity-kernel test (PY6) and all reference comparisons assume out-of-border pixels are 0. A student using edge/reflect padding or `mode='same'` from a library will mismatch the reference on border pixels and fail PY7/PY8 even if the interior is correct.
- **Float output required** — casting back to `uint8` or clipping (natural instinct for "displayable image") breaks the `atol=1e-4` comparisons, especially for Sobel outputs which are signed.
- **Sample tiles vs stated dataset size mismatch:** the question text says 100 tiles / 9 categories, but the conftest samples `tile_001…tile_500.jpg` and its comment references "the full 900-image range." Confirm the actual `images/` folder contents before reusing — the grader only reads the 5 named `SAMPLE_TILES`, so those specific filenames must exist.
- **Library ban is enforced by the reference, not by import-blocking** — nothing stops a student importing cv2/scipy, but a library call that produces different padding/normalization than `_ref_apply` will simply fail the numerical match. The ban is effectively self-enforcing through exactness.

## Dataset status
USED — do not reuse the UC Merced Land Use aerial tiles for another classical-CV / filtering assignment.
