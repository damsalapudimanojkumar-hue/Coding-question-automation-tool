# SVM Classification — Apple Quality (Good/Bad)

**Topic:** SVM Classification (binary)
**Dataset used:** Apple Quality — `apple_train.csv` (3,200 rows × 8 cols), `apple_test.csv` (800 rows × 7 cols)
**Model/algorithm:** `sklearn.svm.SVC` (RBF kernel, `random_state=42`)
**Assignment type:** tabular / classification

## Problem statement summary
Binary classification: predict apple `Quality` (good/bad) from 7 physical property features (Size, Weight, Sweetness, Crunchiness, Juiciness, Ripeness, Acidity — all already pre-scaled/synthetic-looking continuous values in the raw CSV, e.g. negative numbers, not raw physical units). Full pipeline: `LabelEncoder` on the target (`good`/`bad` → 0/1), `StandardScaler` on all 7 features, `SVC(kernel='rbf')` trained on encoded+scaled data, predict on similarly-scaled test data. Classes are well balanced in training (1603 good / 1597 bad) and in the held-out ground truth (413 / 387).

## Test cases used (PY1–PY3)
- **PY1 (structural):** Code runs without error.
- **PY2 (metric threshold):** Accuracy ≥ 0.82. Predictions are found via a fixture that first tries the **exact variable name** `svm_predictions`, and if not found, falls back to scanning all numpy arrays in student namespace for a 1D array of length 800 with values restricted to `{0, 1}` — a structural fallback so the test isn't purely name-dependent, but still benefits from the common-sense naming convention used in the reference solution.
- **PY3 (metric threshold):** Precision (macro-averaged, `average="macro"`) ≥ 0.82. Macro averaging is the deliberate choice here, not binary/weighted — important because it means a model that just predicts the majority class for everything would NOT pass, since macro precision penalizes poor performance on the minority class equally regardless of class size.

## Threshold / calibration notes
- Both accuracy and precision thresholds set at 0.82 — same number for both, reflecting a fairly well-separated dataset where reference `SVC(kernel='rbf')` likely scores comfortably above this with meaningful buffer room.
- **Macro F1/precision choice is intentional and should be the default pattern whenever classes are roughly balanced and a lazy "predict majority class" solution is a real risk** — this is exactly the kind of calibration rule mentioned as a known pitfall in other contexts (e.g. avoid binary-only precision when a trivial predictor could exploit class imbalance or near-balance).

## Gotchas / things that tripped things up or are easy to get wrong
- **No `question.json` was generated for this assignment either** — same gap as the SLR assignment. Only the SGDRegressor/Gradient Descent assignment in this set has a finalized `question.json`.
- The predictions-detection fixture is a strong reusable pattern worth copying for future classification assignments: try the conventional variable name first, fall back to shape+dtype+value-set detection (1D array, correct length, value set is a subset of `{0, 1}`) so the test isn't fragile to variable naming but still rewards following common naming conventions.
- Ground truth (`tests/ground_truth.csv`) stores the target **already label-encoded** as 0/1 (not the original `good`/`bad` strings) — meaning the test pipeline assumes the student's `LabelEncoder` produces the same 0/1 mapping as whatever encoder built the ground truth. Worth flagging as a coupling risk: if a student's `LabelEncoder.fit_transform` happens to map labels in the opposite direction (which `LabelEncoder` would not do here since it sorts alphabetically — "bad" < "good" alphabetically, so bad=0, good=1 is actually guaranteed — but this is exactly the kind of assumption that should be double-checked explicitly for any future assignment using `LabelEncoder` on a target with non-alphabetically-obvious class ordering).
- Feature values in the raw CSV are already on a roughly standardized-looking scale (small numbers, some negative) even though `StandardScaler` is still applied in the reference solution — worth noting if generating a similar synthetic dataset in future, since it could mislead someone into thinking scaling isn't needed when it still meaningfully matters for SVM with RBF kernel.

## Dataset status
USED — do not reuse Apple Quality dataset for another binary classification assignment.