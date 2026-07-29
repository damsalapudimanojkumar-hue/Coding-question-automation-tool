# Code-Editor Config Format

The design agent's job is to emit one **config** dict per question. The generator
(`tools/testcase_generator.py`) turns it into the platform deliverable (a zip with
`questions.json` + `question_sets_questions.json`). You author the config; the
generator computes the outputs by running `solution_code` and encodes the evaluator.

## Fields

**Required**
| Field | Type | Meaning |
|---|---|---|
| `question_text` | str (markdown) | Full problem statement: prose + formula + Parameters + Returns |
| `short_text` | str | Short title |
| `function_name` | str | Exact function name (must match `solution_code` and `starter_code`) |
| `param_names` | list[str] | Ordered parameter names that vary across test cases (omit params that only ever use their default, e.g. `epsilon`) |
| `starter_code` | str | Signature + `# Your code here` + `pass`, with type hints |
| `solution_code` | str | Reference NumPy (or PyTorch) solution — platform reference AND the source of truth for computed outputs |

**Optional (with defaults)**
| Field | Default | Meaning |
|---|---|---|
| `difficulty` | `EASY` | `EASY` / `MEDIUM` / `HARD` |
| `language` | `PYTHON38_DATASCIENCE` | platform language code |
| `time_limit` | `4.0` | seconds per test case |
| `rounding` | `4` | decimal places for output comparison |
| `library` | `numpy` | `numpy` or `pytorch` |
| `rephrased_question_text` | — | if present, used in output instead of `question_text` |
| `rephrased_short_text` | — | if present, used instead of `short_text` |
| `test_definitions` | `[]` | the test cases (below) |

**Each `test_definitions` item**
| Field | Default | Meaning |
|---|---|---|
| `inputs` | required | dict `{param_name: value}` — only inputs; outputs are auto-computed |
| `is_hidden` | `false` | visible vs hidden |
| `weightage` | `10` | score weight (all weights should sum to 100) |
| `tags` | — | optional |
| `display_text` | — | optional student-facing note |

## Hard rules
- `function_name` must be identical in `function_name`, `starter_code`, and `solution_code`.
- `param_names` must be ordered and complete for the values you pass in `inputs`.
- Weightages should sum to **100** (the generator only sums them; it doesn't enforce 100).
- `inputs` carries only inputs — the generator computes each `output` by running `solution_code`, then rounds to `rounding` dp. Grading is exact string match on the rounded output.
- The **evaluator (`main.py`) is boilerplate** — only `FUNC_NAME` (and numpy/pytorch branch) differs. You never write it; the generator emits and base64-encodes it.

## Output shapes the generator handles
scalar float, 1-D list (vector), 2-D list (matrix), tuple (multiple returns), dict. See the design guide for choosing test cases per shape.
