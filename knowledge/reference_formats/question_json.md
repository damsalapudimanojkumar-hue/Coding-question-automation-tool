# question.json — Reference Format

## Structure
Always a JSON **array** containing exactly one question object.
Never a plain dict with a `test_cases` key at the top level.

## Complete Template
```json
[
  {
    "question_id": "<new UUID4>",
    "question_text": "<full markdown problem statement — see markdown_format.md>",
    "short_text": "<one-line title, e.g. 'K-Means Clustering: Wholesale Client Segmentation'>",
    "toughness": "MEDIUM",
    "content_type": "MARKDOWN",
    "question_type": "IDE_BASED_CODING",
    "question_key": "coding_practice_<topic_slug>",
    "question_asked_by_companies_info": [],
    "question_format": "CODING_PRACTICE",
    "ide_session_id": "",
    "test_cases": [
      {
        "test_case_enum": "PY1",
        "display_text": "<student-facing hint if this test fails>",
        "weightage": 10
      },
      {
        "test_case_enum": "PY2",
        "display_text": "<student-facing hint>",
        "weightage": 20
      }
    ],
    "multimedia": [],
    "language": "ENGLISH",
    "solutions_metadata": []
  }
]
```

## Rules
- `toughness` is on the question object ONLY — never on individual test cases
- `weightage` values across all test cases must sum to exactly 100
- `display_text` is student-facing — write as a helpful hint, not a technical assertion
- `question_id` and `ide_session_id` are both freshly generated UUID4 values,
  set by the pipeline after generation (the model just writes placeholder UUIDs)
- The saved file is named `<question_id>.json` (not `question.json`)
- `question_asked_by_companies_info` is always an empty array
- No em dashes anywhere in any text field
- `question_key` uses snake_case with `coding_practice_` prefix

## question_text Markdown Format
The `question_text` field contains full markdown. Follow this structure:
1. `## Title` — business scenario (2-3 sentences)
2. `---`
3. `### Dataset` — table of columns with descriptions ONLY. Do NOT include
   train/test row counts or sample-size details (e.g. "Training set: 4000 rows").
4. `---`
5. `### Tasks` — numbered list, outcome-based not step-prescriptive; keep tasks
   general (state what to achieve, not exact hyperparameters)
6. `---`
7. `### Expected Output` — what variables/shapes the student must produce
8. `### Important Instructions` — always last; state that imports and data loading are pre-filled

### Markdown spacing (renders cleanly in preview)
Put a blank line between every block: between each numbered/bulleted item, before
and after tables, and around every `---`. Do not pack list items on consecutive
lines with no blank line between them.

## Dataset Display Rule

- For 8 or fewer columns, show the Markdown table directly.
- For more than 8 columns, wrap the table in a `<details>` block using the blue-button summary style from `question_text_patterns.md`.
- For image / computer-vision datasets, do not use a column table at all: list the
  class names and the image format (size, channels) and note the folder-per-class
  training layout plus the flat test folder. See `question_text_patterns.md`.

## Test Case Rules

- PY1 is always "The solution should run without any errors." with weight 10.
- Test enums are sequential with no gaps.
- The final test case is the performance threshold.
- Student-facing `display_text` must be helpful and must not reveal the solution.
