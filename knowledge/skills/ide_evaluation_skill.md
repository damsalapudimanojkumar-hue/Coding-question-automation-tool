---
name: ide-evaluation
description: "Create pytest-based evaluation files for IDE coding assignments on the NxtWave/CCBP platform. Use this skill whenever someone asks to create test cases, evaluation, conftest.py, test_solution.py, or any pytest files for coding practice questions. Also use when they mention ccbp submit, IDE-based coding, question loading, requirements.txt for platform, or anything related to evaluating student code submissions. This skill covers DL, NLP, ML, and general Python assignments. Always read this skill before generating any evaluation files."
---

# IDE Evaluation Skill

Generate pytest-based evaluation files for IDE coding assignments on the NxtWave/CCBP learning platform.

## What This Skill Produces

1. `conftest.py` — fixtures to run student code and prepare test data
2. `test_solution.py` — test cases that check student's work
3. `pytest.ini` — pytest configuration
4. `question.json` — test case labels and scoring
5. Ground truth data generation script
6. Local testing setup + platform upload instructions

## When To Use

- User wants to create test cases or evaluation for a coding assignment
- User shares a problem statement and wants evaluation files
- User mentions conftest.py, test_solution.py, pytest, ccbp submit, or IDE evaluation
- User wants to evaluate student code for ML/DL/NLP/Python assignments

---

## Workflow — Follow This Order

### Phase 1: Understand the Assignment

The user provides a **problem statement**. Read it and understand:
- What the student needs to build
- What libraries are involved
- What the expected output looks like

Then ask the user:
- "Do you have a solution for this assignment, or should I generate one?"
- "Which platform session will this run on?" (determines pre-installed libraries)
- "What dataset will students use?"

If the user provides a solution, move to Phase 2.
If the user wants you to generate a solution, write it and get approval before Phase 2.

### Phase 2: Propose Test Cases — COLLABORATE

Analyze the solution step by step. For each critical step, identify what could go wrong.

Then PRESENT your suggested test cases to the user:

```
"Based on the solution, here are the test cases I suggest:

PY1 — Code runs without errors
PY2 — [main output exists]
PY3 — [structure/shape check]
PY4 — [specific logic check]
...

Do you have any specific metrics or checks in mind?
Should I add, remove, or modify any of these?"
```

WAIT for user input. They may:
- Add checks you didn't think of (common student mistakes they've seen)
- Remove checks they don't care about
- Adjust thresholds
- Suggest entirely different evaluation criteria

DO NOT proceed to file generation until test cases are agreed upon.

### Phase 3: Generate Files

Only after test cases are finalized:

1. Generate all files and present in chat for **local testing first**
2. Provide the flat directory structure for local testing
3. After user confirms local tests pass, provide platform zip instructions

### Phase 4: Local Testing Support

If user encounters errors during local testing, help debug.
Read `references/errors.md` for known issues and fixes.

### Phase 5: Platform Upload

After local testing passes, instruct user on creating the two zips.
Read `references/platform.md` for the exact structure.

---

## CRITICAL DESIGN RULES

### Test Detection — Find by TYPE, Not by Name
Never depend on variable names. Scan student_code by isinstance and structure.
Read `references/detection_patterns.md` for all detection patterns.

Exception: When unavoidable (word2idx, tokenizer), try name first with type fallback,
AND tell students the exact variable name in the question description.

### Fixture Rules
- All fixtures: `scope="session"` (student code runs once)
- Never use `pytest.fail()` inside fixtures — store error, let tests handle it
- Always use `encoding="utf-8"` in open() for Windows compatibility
- File paths relative to pytest root: `"tests/ground_truth.csv"`

### No Hooks
Do NOT add `pytest_runtest_makereport` or `pytest_sessionfinish` hooks to conftest.py.
The platform generates its own results.json and overwrites anything custom.
Keep conftest.py clean — fixtures only.

### Solution Rules
- Starter file is `solution.ipynb` (Jupyter notebook), NOT `solution.py`
- The starter notebook follows a fixed 3-cell structure:
  1. **Markdown cell:** Title + brief description + all tasks with expected variable names
  2. **Code cell:** All imports + data loading (with comment: `# Run this cell before writing your solution`)
  3. **Code cell:** Just `# Write your code here` (no variable placeholders - expected variables are already listed in the markdown cell, and placeholders restrict students to a single cell)
- Do NOT use em dashes in starter notebooks
- Never include `!pip install` in the starter notebook — libraries go in requirements.txt only
- No matplotlib `plt.show()` in solution code for platform (blocks exec)

### Platform Rules
- requirements.txt ONLY in tests zip (next to pytest.ini) — prevents student edits
- test_type must be PYTHON (DATASCIENCE doesn't support pytest)
- Changing test_type requires NEW session ID
- Assert messages are never shown to students — use question.json display_text

---

## Reference Files — Read Before Generating

| File | When to Read | What It Contains |
|------|-------------|-----------------|
| `references/platform.md` | Before any file generation | How ccbp submit works, server merge, results.json, session types |
| `references/templates.md` | When generating files | conftest.py, test_solution.py, pytest.ini, question.json templates |
| `references/detection_patterns.md` | When designing fixtures | 14 patterns for finding objects by type instead of variable name |
| `references/dependencies.md` | When writing requirements.txt | Pre-installed libs per session, torch issue, requirements.txt rules |
| `references/errors.md` | When user hits errors | All known errors: pytest, platform, Windows, encoding, file paths |
| `references/patterns.md` | When designing test cases | Test ideas by category, thresholds, time limits, ground truth generation |