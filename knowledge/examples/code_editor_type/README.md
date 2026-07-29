# Worked Examples — code_editor_type

Gold **config** examples for function-based (code-editor) questions. These are the
*input configs* the design agent learns from — NOT the generated output zips (those are
bloated with a base64 evaluator and teach nothing).

The loader feeds 1-2 of these to the code-editor design agent as the format+quality
reference. Keep the set small (no redundancy).

## What's here
- `bce_config.json` — Binary Cross-Entropy. **Scalar (float) output.** Verified: running
  the generator on it reproduces the known platform outputs exactly.

## What to add (you)
Drop more gold **config** files here to cover output shapes the agent should learn:
- a **1-D vector output** example (e.g. `linear_regression_config.json`, softmax) — paste your canonical config.
- a **matrix (2-D) output** example, if you'll have such questions.
- a **tuple / multiple-return** example, if applicable.

## Format
Each file is a config dict exactly as in
`knowledge/reference_formats/code_editor_type/config_format.md`:
required `question_text`, `short_text`, `function_name`, `param_names`, `starter_code`,
`solution_code`; plus `test_definitions` (inputs only — the generator computes outputs).

`README.md` is ignored by the loader.
