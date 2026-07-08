# Platform Rules

## Session Types
| Session | Pre-installed | Notes |
|---|---|---|
| DSML_DL | torch, torchmetrics, accelerate, scikit-learn, pandas, numpy | Best for DL assignments |
| DSML_NLP | nltk, spacy, regex, wordcloud | torch NOT pre-installed here |

## requirements.txt
- Contains ONLY `pytest` for DSML_DL / DSML_NLP sessions (everything else
  pre-installed)
- Lives ONLY in tests.zip, next to pytest.ini -- never in prefilled_codes.zip
  (prevents students from editing it)

## test_type
- Must be `PYTHON` in the loading sheet, never `DATASCIENCE`
  (DATASCIENCE doesn't support pytest)
- Changing test_type requires a NEW session ID -- cannot change in place

## question.json identifiers
- `question_id` and `ide_session_id` are both freshly generated UUID4 values,
  produced by the pipeline (Python `uuid` library) after the file is written --
  never left empty, never trusted to model-typed values.
- The deliverable question file is named `<question_id>.json`, so the filename
  matches the id inside it.

## Forbidden in solution.ipynb (causes platform failures)
- `!pip install` -- Jupyter-only syntax, causes SyntaxError in exec()
- `plt.show()` -- blocks execution on platform, remove or guard with `if` block
- `/content/` paths -- Colab-only, use relative paths instead

## Dataset hosting
- Amazon S3: `https://s3.ap-south-1.amazonaws.com/new-assets.ccbp.in/frontend/content/aiml/`
- Datasets should be loaded via `urllib.request` in starter notebooks when
  hosted externally
- Dataset size: under 10 MB
- Row count: 300-400+ samples where feasible
- Must differ meaningfully from session material (not copy-paste-able)

## Image / Computer-Vision datasets
- Ship REAL image files, never a CSV of pixel values.
- Training data: `dataset_train_<CODE>/<class_name>/<images>` — one subfolder per
  class, folder name = the exact class label (torchvision `ImageFolder` layout).
- Test data: `dataset_test_<CODE>/<images>` — a flat folder of UNLABELED images the
  student predicts.
- Ground truth: `tests/ground_truth_<CODE>.csv` with two columns `filename,label`
  (the ONLY CSV). The grader matches predictions to labels by filename.
- Keep it small and fast: a few hundred images total, downscaled (e.g. 32x32 to
  128x128), classes roughly balanced, under ~15 MB zipped.
- Runs on the DSML_DL session (torch/torchvision pre-installed). Students load the
  training folder with `datasets.ImageFolder`; no S3, no downloads in the notebook.

## opencv
Use `opencv-python-headless`, not `opencv-python`, on headless eval servers.

## Threshold calibration
- Build a genuinely good reference solution first
- Measure its real performance (RMSE, accuracy, F1, etc.)
- Set threshold ~40-50% buffer below reference performance
- Use macro F1 (not binary/micro) when lazy all-ones predictors are a risk
- Test thresholds against multiple solution variants before finalizing,
  not just the one reference solution

## Lazy solution validation (mandatory QA step)
Always verify a lazy/wrong solution actually FAILS the appropriate test
cases -- not just that the reference solution passes. A test suite that
only checks the reference solution passing is not sufficient validation.

## Assert messages
Assert messages in test_solution.py are never shown to students. Any hint
or guidance for students must go in `question.json` -> `display_text`,
written as a helpful nudge, not a technical assertion message, and without
revealing the solution approach.
