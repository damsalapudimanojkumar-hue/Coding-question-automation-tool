# Question Description Structure (code-editor)

How a code-editor question's `question_text` should be written, distilled from the
reference examples (accuracy_score, KNN, BCE). The design agent must follow this.

## Section order (lean, in this order)

1. Intro (1-2 sentences): what the function computes and, briefly, why it matters.
   Start directly with the sentence, e.g. "Write a Python function that ...".
2. Formula (plain text - see below). Introduce it with a short line like "The X formula is:".
3. `**Parameters**` - a bullet per parameter: name, type, meaning.
4. `**Requirements**` (optional) - bullets stating what to compute / constraints
   (e.g. "Do not use sklearn's accuracy_score function", "clip to avoid log(0)").
5. `**Returns**` - the return type, rounding, and any special-case behavior.
6. `**Examples**` (optional, added automatically) - 1-2 concrete input -> output pairs.

## Hard rules

- NO title heading. Do NOT begin with `#`/`##` and a title. The title is the separate
  `short_text` field; the description starts with the intro sentence.
- Use bold labels `**Parameters**`, `**Requirements**`, `**Returns**` - NOT `##` headings.
- Keep it lean. Do not invent extra `###` subsections; the four labels above are enough.
- One concept per question. Test the concept, never the source artifact ("in the notebook").

## Formula style (the platform does NOT render LaTeX)

Write every formula in PLAIN TEXT. The platform's markdown renderer does not support LaTeX,
so `$...$` / `$$...$$` and `\frac`, `\text`, `\sum`, `\sqrt`, `\begin{}` etc. appear as raw
source and look broken.

- Use words and simple symbols: `=  /  *  +  -  ^` and, where helpful, Unicode:
  `Σ  √  ×  ·  ≤  ≥  π  →`.
- House style (from the references):
  - accuracy: `Accuracy = Number of correct predictions / Total number of predictions`
  - a sum: `MSE = (1/N) * Σ (y_true - y_pred)^2`
  - a norm: `||u|| = √(Σ u_i^2)`
- For a formula too complex for plain text, put a markdown image placeholder on its own line
  and describe it in words:
  `![binary cross-entropy formula](PLACEHOLDER)`
  (a person resolves the placeholder to an S3 image URL later, exactly like the BCE example).

## Parameters / Returns formatting

```
**Parameters**

- `y_true` (list[int]): True labels
- `y_pred` (list[int]): Predicted labels

**Returns**

- float: The accuracy score, rounded to four decimal places
```

Match the declared return type in the solution (integer labels -> cast to int so outputs
are `1`, not `1.0`).

## Examples section (auto-generated)

When enabled, an `**Examples**` section is appended using the first 1-2 VISIBLE test cases
with their real computed outputs. It is written as a **python code block** with named
arguments (LeetCode-style), so long argument lists stay readable and don't wrap in a narrow
panel:

    **Examples**

    ```python
    relu(x=[1.0, -1.0, 0.0])
    # -> [1.0, 0.0, 0.0]
    ```

For functions with many parameters, each argument goes on its own line:

    ```python
    lstm_cell_forward(
        x_t=[1.0],
        h_prev=[0.0],
        c_prev=[0.0],
    )
    # -> ([0.1743], [0.2876])
    ```

The design agent does not write this section itself (it runs before the test outputs exist);
it is added from the computed visible cases so the shown examples always match grading.
