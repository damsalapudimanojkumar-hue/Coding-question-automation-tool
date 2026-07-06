# Worked Examples — vscode_type

One **subfolder per real, shipped assignment**. Each subfolder holds the three
files that form the grading contract, as real files (no markdown wrapping):

```
examples/vscode_type/
  bagging_forest_cover/
    question.json
    conftest.py
    test_solution.py
  regularization/
    question.json
    conftest.py
    test_solution.py
```

The loader reads every subfolder here and feeds the bundles to the
file-generating agent (Agent 3). These teach it what a good finished
assignment looks like. Rules live in `knowledge/skills/`; these are examples,
not rules.

## What to do

1. Make a folder named after the assignment (snake_case, e.g. `svm_apple_quality`).
2. Drop exactly these three files in it, named exactly:
   - `question.json`
   - `conftest.py`
   - `test_solution.py`
3. That's it. `solution.ipynb`, `pytest.ini`, `requirements.txt` are not needed
   (trivial or derivable from the skill guidelines + question.json).

Only these three filenames are picked up. This `README.md` is ignored.

## Keep the contract aligned across the three files

`question.json` `PY2` (display_text + weightage)
  ↔ `test_solution.py` `test_PY2`
  ↔ the `conftest.py` fixture that test depends on.

Enums sequential, weightages sum to 100.

## Note on routing (later)

For now every example is loaded on every run. Once there are more than ~5,
we will filter by task type (classification vs regression) so irrelevant
examples don't bloat the prompt. At that point add a small `meta.json` per
folder (`{"assignment_type": "tabular", "task_type": "classification"}`) or we
infer it from the files.
