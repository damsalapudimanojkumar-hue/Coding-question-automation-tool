# Gradient Descent — SGDRegressor (Power Plant Energy Output)

**Topic:** Gradient Descent / SGDRegressor
**Dataset used:** Power Plant (CCPP-style synthetic) — `power_plant_train.csv`, `power_plant_test.csv`
**Model/algorithm:** `sklearn.linear_model.SGDRegressor`
**Assignment type:** tabular / regression

## Problem statement summary
Business framing: a power grid operator needs to predict net hourly energy output of a combined cycle power plant from environmental sensor readings (Temperature, Exhaust_Vacuum, Ambient_Pressure, Relative_Humidity), to decide how many peaker plants to activate. Session context: students implemented Gradient Descent from scratch in the session; this assignment bridges that to `SGDRegressor` as the sklearn equivalent. Task: load data, handle missing values, apply `StandardScaler` (explicitly called out as required — SGD won't converge without it), train `SGDRegressor` on all four features, predict on test data using the same fitted scaler.

Dataset shape: train ~(rows with some nulls) with 4 features + `Energy_Output` target; test has 4 features only, predictions checked against a separate `ground_truth.csv` with the true `Energy_Output` column.

## Test cases used (PY1–PY4)
- **PY1 (10):** Code runs without error.
- **PY2 (20):** A trained `SGDRegressor` instance is found in student namespace via `isinstance(v, SGDRegressor)` — not by variable name.
- **PY3 (30):** Model learned correct feature relationships — checks `len(coef_) == 4` (all features used), each coefficient's **sign** matches expected direction (`[-1, -1, +1, -1]` for Temperature, Exhaust_Vacuum, Ambient_Pressure, Relative_Humidity), and `intercept_` falls in range `[440, 470]`.
- **PY4 (40):** RMSE on test data ≤ 5.5. Scaler is auto-detected by `isinstance(v, StandardScaler) and hasattr(v, "mean_") and len(v.mean_) == 4` — if found, test data is transformed through it before scoring; if not found, raw test data is used as fallback so the test doesn't hard-fail on missing scaler detection.

## Threshold / calibration notes
- RMSE ≤ 5.5 — set from the reference solution's actual test RMSE with buffer room.
- Intercept range [440, 470] is a tight but real range observed from the reference `SGDRegressor` fit — not arbitrary; reflects the actual scale of `Energy_Output` in this dataset (around mid-400s MW).
- Coefficient **sign** check (not magnitude) is the chosen way to verify "correct feature relationships" — robust to different random seeds / minor hyperparameter differences while still catching a fundamentally wrong model (e.g. unscaled features, wrong target, or features swapped).

## Gotchas / things that tripped things up or are easy to get wrong
- Detecting the scaler is non-trivial: must check it's actually a 4-feature `StandardScaler` (`len(v.mean_) == 4`) so it doesn't accidentally pick up a scaler used for something else in the same notebook namespace.
- PY3's sign-check approach (rather than exact coefficient value matching) is the right level of strictness for SGD-based models — exact values vary run to run even with `random_state` set, because SGD is iterative; signs and intercept range are stable enough to assert on, magnitudes are not.
- This is one of the few assignments with a full `question.json` already finalized including `toughness: MEDIUM` and a collapsible `<details>` hint block showing the exact `SGDRegressor` + `StandardScaler` code pattern — useful as the template for how much scaffolding/hint code to give students directly in the problem statement.
- No time-limit test in this one (per project's later updated rule that not every assignment needs one).

## Dataset status
USED — do not reuse Power Plant dataset for another tabular regression assignment.