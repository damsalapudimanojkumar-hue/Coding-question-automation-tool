# Inception CNN — TrafficVision Traffic Sign Classifier (Computer Vision Assignment 2)

**Topic:** GoogLeNet/Inception module (single-model, focused)
**Dataset used:** Traffic Signs — `traffic_signs.npz` (32×32×3 RGB, 10 classes), hidden `tests/ground_truth.npz`
**Model/algorithm:** PyTorch `nn.Module` with at least one custom `InceptionModule`
**Assignment type:** image / multi-class classification (deep learning, PyTorch)

## Problem statement summary
Build an Inception-inspired CNN to classify 32×32 RGB traffic-sign images into 10 categories. Task: load `traffic_signs.npz`, build `Dataset`/`DataLoader`, implement a custom `InceptionModule` (`nn.Module`) with four parallel branches — (1) 1×1 conv; (2) 1×1 bottleneck → 3×3; (3) 1×1 bottleneck → 5×5; (4) 3×3 MaxPool → 1×1 — concatenated along channels, then a classifier with ≥1 Inception module ending in FC layers for 10-class output, trained with `CrossEntropyLoss` to ≥70% test accuracy. Single expected variable: `model` (`nn.Module`).

This is the **focused, single-model version** of the Inception task also covered as Task 4 of **[[cnn_architectures_waste_sorting]]** (CV Assignment 1). Here Inception is the whole assignment, not one of three architectures, and the accuracy bar is much higher.

## Test cases used (PY1–PY6)
- **PY1 (5):** Code runs without error.
- **PY2 (10):** An `nn.Module` instance exists in student namespace.
- **PY3 (25):** Model uses ≥3 distinct `Conv2d` kernel sizes (e.g. 1×1, 3×3, 5×5) — proves multi-scale/Inception structure. Highest weightage.
- **PY4 (20):** Model has ≥3 `Conv2d` layers with `kernel_size=1` (bottleneck reduction).
- **PY5 (15):** Model outputs 10 units on a `(1,3,32,32)` probe.
- **PY6 (25):** Hidden-test accuracy ≥ 0.70. Also highest weightage — this assignment genuinely requires a working, well-trained model.

## Threshold / calibration notes
- **Accuracy ≥ 0.70 is a much tighter bar than Assignment 1's 0.40** — because this is a single focused model (student can spend the whole budget training it well) and traffic signs are a more separable / less noisy category set than mixed waste photos. This is the deliberate calibration difference between the two Inception assignments: Assignment 1 grades "did you build three architectures" (loose accuracy floor), Assignment 2 grades "did you build *and train* one Inception model well" (real accuracy bar).
- PY3 (multi-scale kernels, 25) + PY6 (accuracy, 25) together are half the weight — architecture correctness and actual performance are weighted equally.
- **No explicit time-limit test** in this assignment (unlike Assignment 1's PY11 / SLR set) — per the project's later rule that not every assignment needs a time-limit test.

## Gotchas / things that tripped things up or are easy to get wrong
- **Model detection is "most parameters wins"** — the `model` fixture scans all `nn.Module` instances and picks the one with the largest `sum(p.numel())`. If a student leaves a small helper module (e.g. a bare `InceptionModule` instance) in the namespace alongside the full classifier, the full classifier normally wins by param count — but a student who instantiates a huge sub-module and a small classifier could trip this. Generally robust; worth knowing it is *not* name-based.
- **`model_conv_kernel_sizes` compares `m.kernel_size` tuples directly** — PyTorch stores `Conv2d` kernel size as a tuple like `(3, 3)`, so a student using rectangular kernels or `nn.LazyConv2d` variants could produce unexpected distinct-size counts. Standard square kernels behave as expected.
- **PY3 counts *distinct* sizes, PY4 counts *number* of 1×1 layers** — a student could pass PY3 with 1×1/3×3/5×5 present but fail PY4 if they used fewer than 3 bottleneck 1×1 convs (e.g. omitted the 1×1 before the MaxPool branch, or shared one bottleneck). The four-branch spec naturally yields ≥4 1×1 convs, so following the branch layout passes both.
- **Bug in `test_solution.py`'s `_skip_if_broken(student_code, model)` guard:** the second check is `if model is not None and model is None:` — a contradiction that is always `False`, so the "no model found" skip never fires from this helper. In practice PY2's assert catches a missing model first and downstream tests still work because `model is None` propagates through fixtures, but the guard is dead code. Worth fixing (`if model is None: pytest.skip(...)`) if this conftest is reused as a template.
- Preprocessing contract in the grader: `torch.FloatTensor(X_test).permute(0,3,1,2) / 255.0`. As with Assignment 1, a student training on a different normalization scheme will underperform on PY6 regardless of architecture.

## Dataset status
USED — do not reuse the Traffic Signs (`traffic_signs.npz`) dataset for another image-classification assignment.
