# CNN Architectures — Smart Waste Sorting (Computer Vision Assignment 1)

**Topic:** CNN architecture evolution (LeNet-5 → AlexNet-style → GoogLeNet/Inception)
**Dataset used:** Waste Images — `waste_images.npz` (3,200 train + 800 validation, 32×32×3 RGB, 10 classes), hidden `tests/ground_truth.npz`
**Model/algorithm:** PyTorch `nn.Module` CNNs — three of them (LeNet-5, an improved ReLU+Dropout variant, and an Inception-module network)
**Assignment type:** image / multi-class classification (deep learning, PyTorch)

## Problem statement summary
Business framing: a recycling company automates waste sorting by classifying a photo of a waste item into one of 10 categories (cardboard, glass, metal, paper, plastic, trash, organic, battery, clothes, electronics). Students walk through the *history of CNN architectures* on the same dataset: build LeNet-5 (`lenet_model`), then improve it with AlexNet ideas — `nn.ReLU` + `nn.Dropout` (`improved_model`), then build a GoogLeNet-inspired network with a custom Inception module (`inception_model`) whose four parallel branches (1×1; 1×1→3×3; 1×1→5×5; 3×3 MaxPool→1×1) are concatenated along the channel dimension. Each model trains 10 epochs with `CrossEntropyLoss` + Adam. Finally students evaluate, build a confusion matrix, and store the three validation accuracies in a `model_results` dict.

Data contract: images scaled to `[0, 1]`, channel-first `(N, 3, 32, 32)`, labels integers 0–9. Custom `Dataset` + `train_loader`/`val_loader` required.

## Test cases used (PY1–PY11)
- **PY1 (5):** Code runs without error.
- **PY2 (10):** At least 2 `DataLoader` objects exist, and among the datasets they wrap the sizes must include exactly 3200 (train) and 800 (val) — detected structurally by `len(loader.dataset)`, not by variable name.
- **PY3 (10):** `lenet_model` has ≥2 `Conv2d` + ≥2 `Linear` layers and outputs 10 class scores on a `(1,3,32,32)` probe.
- **PY4 (10):** `lenet_model` hidden-test accuracy ≥ 0.40.
- **PY5 (10):** `improved_model` uses `nn.ReLU` and at least one `nn.Dropout`.
- **PY6 (10):** `improved_model` hidden-test accuracy ≥ 0.40.
- **PY7 (15):** `inception_model` contains ≥3 `1×1` convs plus a `3×3` and a `5×5` branch and a `MaxPool2d` branch, and outputs 10 classes. Highest weightage — the Inception module is the centerpiece.
- **PY8 (10):** `inception_model` hidden-test accuracy ≥ 0.40.
- **PY9 (10):** A 10×10 confusion matrix is present, computed on the full 800-sample validation set — verified structurally (`np.ndarray`/`Tensor`/`DataFrame`, shape `(10,10)`, non-negative numeric, entries sum to exactly 800).
- **PY10 (5):** `model_results` is a dict of 3 float accuracies in `(0, 1]`.
- **PY11 (5):** Runs within 600 seconds.

## Threshold / calibration notes
- **Accuracy thresholds are deliberately loose at 0.40** for all three models — this is a 10-class problem (random = 0.10) trained for only 10 epochs on 32×32 images, so 0.40 comfortably clears "the model actually learned something" without requiring a strong fit or GPU-scale training. The goal is *architecture comprehension*, not squeezing accuracy.
- **600s time limit** is generous because three separate models each train 10 epochs on CPU inside the grader.
- Architecture checks (PY3/PY5/PY7) carry as much or more weight than accuracy — the assignment grades *whether the right architecture was built*, with accuracy as a floor that proves it was actually trained.

## Gotchas / things that tripped things up or are easy to get wrong
- **Model detection is by architectural signature, not variable name** — the fixtures first try the conventional name (`lenet_model`, `improved_model`, `inception_model`) then fall back to distinguishing features: LeNet = has 5×5 kernel, zero 1×1 convs, no dropout; improved = zero 1×1 convs *with* dropout; inception = ≥3 1×1 convs plus 3×3 and 5×5. This signature-based fallback is the key reusable pattern for multi-model assignments, but it means the three fallbacks must stay mutually exclusive — if a student's "improved" model happens to use a 5×5 kernel and no dropout it could be misdetected as LeNet. Naming the variables correctly avoids all ambiguity.
- Every candidate model is first filtered through a **behavioral probe** (`_outputs_10_classes`): it must accept `(1,3,32,32)` and return 10 logits. A model that crashes on that input shape is silently invisible to detection and the test reports "model not found" rather than an architecture failure.
- The `ground_truth` fixture applies the *same* preprocessing students are told to use (`permute` to channel-first, `/255.0`). A student who trains on a *different* normalization (e.g. mean/std standardization) will score poorly on PY4/6/8 even with a correct architecture — the preprocessing contract is implicit but load-bearing.
- Confusion-matrix detection requires entries to sum to **exactly 800** — a matrix built on the train set, a subset, or with normalized/proportion values (summing to ~1 or ~10) will not be detected.
- `improved_model` accuracy threshold (0.40) is the same as LeNet's, so the assignment does **not** actually require the improvement to beat LeNet — it only requires ReLU+Dropout to be present and the model to clear the floor. Worth noting if the intent was to force a measurable improvement.

## Relationship to CV Assignment 2
This assignment's Inception task overlaps heavily with **[[inception_traffic_signs]]** (CV Assignment 2), which is Inception-only on a near-identical 32×32×3 / 10-class contract. If both ship in the same set, the Inception module is taught twice — reuse is intentional (progression) but the datasets must stay distinct (waste images vs traffic signs) to avoid a student copying one solution into the other.

## Dataset status
USED — do not reuse the Waste Images (`waste_images.npz`) dataset for another image-classification assignment.
