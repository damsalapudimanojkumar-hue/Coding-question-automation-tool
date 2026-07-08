# question_text Patterns - Reference

## Business scenario patterns

- Start with "A **[role or organization]** wants to" or "needs to".
- Give real-world stakes in one or two sentences.
- Bold the key role or concept.
- Transition directly into the task.

Good openings include a power grid operator predicting energy output, a health research organization predicting life expectancy, or a telecom company identifying customer churn.

Avoid starting with "In this assignment", referring to what was taught, prescribing unnecessary implementation details, using em dashes, or showing `plt.show()` or `!pip install`.

## Tasks

- Keep tasks general and outcome-based: state what the student must achieve,
  not a step-by-step recipe.
- Do NOT prescribe exact hyperparameters or arbitrary choices (specific
  `n_estimators` lists, `random_state` values, exact estimator counts, etc.).
  Those belong to the student's approach.
- Performance thresholds are the exception: state them explicitly.
- Keep the list short; one clear outcome per point.

## Dataset display rule

- Describe columns/features only. Do NOT include train/test row counts or
  sample-size details (no "Training set: 4000 rows | Test set: 1000 rows").
- 8 or fewer columns: show a plain Markdown table.
- More than 8 columns: use this collapsible block:

### Image / Computer-Vision datasets

For CV assignments the dataset is real image files, not a table of columns. In the
Dataset section, instead of a column table:

- List the exact CLASS names (these are the training subfolder names).
- State the image format: size and channels (e.g. "64x64 RGB").
- Say the training images are provided as one folder per class (ImageFolder
  layout) and the test images as a flat folder to predict.
- Do NOT list pixel columns and do NOT give per-class image counts.
- Use a `<details>` block for the class list when there are many classes.

```html
<details>
<summary style="display:inline-block;padding:8px 16px;background:#3b82f6;color:white;border-radius:6px;cursor:pointer;font-weight:bold;">Learn about the Dataset Features</summary>

| Column | Description |
|--------|-------------|
| `column_name` | Description |

</details>
```

## Expected Output

Always include this section. If exact variable names are required, show them in a table. Otherwise state that the solution must contain a trained model and test-data predictions.

## Important Instructions

This is always the final section. Always say: "The data loading and required libraries are **pre-filled** in your notebook. Do **not** modify them." Add only assignment-specific critical constraints after it.

## Markdown spacing (renders cleanly in preview)

Put a blank line between every block, everywhere in the question_text, not just
between task items:

- a blank line between each numbered or bulleted item
- a blank line before and after every table
- a blank line around every `---` separator
- a blank line between a heading and the text under it

Never pack list items or paragraphs on consecutive lines with no blank line
between them, or the preview renders them as one run-on block.
