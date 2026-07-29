# Test-Case Design Guide (code-editor)

The heart of a good code-editor question is the **test cases** — they define what a
correct answer is, and the hidden ones are what catch wrong implementations. Design
them from the **mathematical formula**, not from any one implementation.

## Rules
- **8-12 test cases** per question.
- **First 3-4 visible** (`is_hidden: false`), the rest **hidden** (`is_hidden: true`).
- **Weightages sum to 100.**
- Outputs rounded to **4 dp** (unless the question says otherwise).
- Design cases so that **any correct implementation passes** and **common wrong ones fail**.
- Compute expected outputs **from the formula** and hand-trace at least 2 — this is your
  check that `solution_code` is correct (the generator computes the authoritative
  outputs by running `solution_code`; the two must agree).

## The four categories

**Category 1 — Basic / example (VISIBLE)**
- The worked examples from the problem statement; simple cases a student can verify by hand.

**Category 2 — Edge (HIDDEN)**
- All-zero inputs; identity matrices / unit vectors; single element; negatives;
  very large values (saturation for tanh/sigmoid); very small / near-epsilon values.

**Category 3 — Dimensional variations (HIDDEN)**
- Minimum valid dimensions; non-square where applicable; larger (3x3, 4x5).

**Category 4 — Domain-specific (HIDDEN)**
- Cases that catch the common wrong approach for THIS concept, e.g. linearly dependent
  data for regression, uniform distribution for softmax, exactly-0/1 probabilities for
  a loss that needs clipping.

## Common mistakes to target with hidden cases
- Off-by-one errors
- Wrong matrix transpose / wrong axis (row vs column)
- Missing bias term
- Not handling division-by-zero / no numerical-stability clipping
- Overflow (e.g. softmax without subtracting the max; BCE without clipping)

## Worked illustration — BCE (see `examples/code_editor_type/bce_config.json`)
Each hidden case targets a specific failure mode:

| input `[y_true, y_pred]` | output | what it catches |
|---|---|---|
| `[1,0,1,0]/[0.9,0.1,0.8,0.2]` (visible) | 0.1643 | the worked example |
| `[1,1,1]/[1,1,1]` (visible) | 0.0 | perfect prediction baseline |
| `[0,0,0]/[0,0,0]` | 0.0 | the label-0 branch |
| `[1]/[0.5]`, `[0]/[0.5]` | 0.6931 | single sample = -log(0.5) |
| `[1,0]/[0.01,0.99]` | 4.6052 | confidently WRONG -> big penalty |
| `[0,0,0,0]/[0.5,...]` | 0.6931 | uniform baseline |
| `[1,0]/[0.0,1.0]` | 34.5392 | **the trap**: prob exactly 0/1 -> without epsilon-clipping this is log(0) -> crash/inf |

The last row is the point of the whole question (the clipping requirement) expressed as a
single hidden case. Aim for at least one such "trap" per likely mistake.
