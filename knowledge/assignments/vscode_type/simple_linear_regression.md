# Simple Linear Regression — Calories Burned Prediction

**Topic:** Simple Linear Regression (single feature)
**Dataset used:** Exercise/Calories — `exercise_train.csv` (12,105 rows), `exercise_test.csv`
**Model/algorithm:** `sklearn.linear_model.LinearRegression`
**Assignment type:** tabular / regression

## Problem statement summary
Single-feature regression: predict `Calories` burned from `Duration` of exercise. Training data has real null values (300 nulls in `Duration`, 225 in `Calories`, out of 12,105 rows) that must be dropped before fitting. Test data (`exercise_test.csv`) has only the `Duration` column — student predicts `Calories`, which is checked against a separate `ground_truth.csv` of 2,895 rows containing both `Duration` and the true `Calories`.

This is intentionally the simplest assignment in the set — single feature, no scaling required (`LinearRegression` via OLS doesn't need it), straightforward `dropna()` for null handling. Used as an early/foundational assignment in the regression sequence, likely preceding the Gradient Descent/SGDRegressor assignment.

## Test cases used (PY1–PY3)
- **PY1 (no explicit weight tag in this file, structural):** Code runs without error.
- **PY2 (structural):** A `LinearRegression` instance is found in student namespace via `isinstance(val, LinearRegression)`, skipping any `__`-prefixed internal fixture keys — not by variable name.
- **PY3 (metric threshold):** RMSE on hidden ground truth ≤ 22. Predictions are generated inside a fixture (`predictions`) by calling the detected model's `.predict()` directly on `ground_truth[["Duration"]]` — i.e. the test pipeline does its own prediction call rather than looking for a pre-computed predictions variable in student code. This is a different pattern from the SVM and Gradient Descent assignments, where predictions are read from student-namespace variables.

## Threshold / calibration notes
- RMSE ≤ 22 reflects the actual scale of `Calories` values in this dataset (values range broadly, single-digit to several hundred) — 22 is a meaningful buffer above the reference `LinearRegression` fit's RMSE, not a tight threshold, appropriate for a foundational single-feature assignment where the goal is "did you correctly build and fit a simple linear model," not precision tuning.

## Gotchas / things that tripped things up or are easy to get wrong
- **No `question.json` was generated for this assignment** — unlike Gradient Descent, this bundle only has the local testing files (conftest, test_solution, ground_truth). Worth checking before reusing this as a template if `question.json` is required.
- This is the only one of the three reference assignments where the test suite **calls `.predict()` itself** on the detected model rather than reading a predictions array from student code. Cleaner for single-feature regression since there's only one obvious way to call `.predict()` and no risk of the student computing it differently; this pattern would NOT generalize well to assignments with multiple plausible prediction-generation steps (e.g. classification with thresholding, or multi-feature pipelines with scaling — see SVM and Gradient Descent assignments for that case).
- Real-world nulls in training data (300 + 225 rows) make this a good "minimum viable" preprocessing assignment — tests implicitly require the student to handle nulls correctly (a broken `dropna()` or skipped null-handling would still likely produce a model, just a worse-fit one, so PY3's RMSE threshold is partly what's actually verifying preprocessing was done, not just PY1/PY2).

## Dataset status
USED — do not reuse Exercise/Calories dataset for another simple regression assignment.