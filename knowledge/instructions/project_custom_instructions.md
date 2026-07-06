# Project Custom Instructions

## Communication preferences
- Short, direct, copy-paste-ready format
- Present options with clear framing for quick decisions
- Detailed structured tables for workflow/content planning
- Seek approval on problem statement + test cases before file generation
- No em dashes anywhere in generated content

## Assignment build workflow (applies across all courses)
1. Review session materials to understand taught concepts and code style
2. Select a differentiated dataset/task that requires the target algorithm
   without being Googlable or copy-pasteable from session
3. Present complete problem statement + test cases for approval before
   generating any files
4. Build reference solution -> calibrate thresholds -> verify reference
   passes all tests
5. Verify lazy/wrong solution fails appropriate tests
6. Generate all four deliverables
7. Run clean-room extraction test before delivery

## Problem statement standards
- Easily readable: bullet points and tables where helpful
- No em dashes anywhere
- Collapsible `<details>` blocks for long dataset feature tables
- Class names with underscores wrapped in backticks to prevent italic
  rendering bugs (e.g. `SGD_Regressor`)
- Shorter version for starter notebook; fuller version for question.json
- No references to session content or what students "learned in session"
- Outcome-based over step-based: don't over-prescribe implementation,
  students get freedom in approach while staying auto-gradable

## Test case design standards
- No time-limit tests unless explicitly confirmed by user
- No redundant checks -- merge related assertions into a single focused test
- Complexity proportionate to question simplicity
- No solution pipeline details in hints/notes
- Evaluation thresholds stated explicitly in the problem statement

## Key principles
- Teaching notebooks (session material) use inline concept-first code, no
  def statements. Function-based structure belongs in assignments only.
- Dataset must differ meaningfully from session material to prevent
  copy-paste.
- Algorithm-dataset coupling: for algorithm-specific tasks (Siamese
  Networks, LoRA/PEFT, KNN, Apriori), the dataset must naturally require
  the specific algorithm taught -- not just any dataset that happens to fit.
- Accuracy vs honesty in audits/reviews: distinguish confirmed errors from
  observations or speculation, don't overstate findings as facts.

## Datasets reserved / already used
See `../assignments/vscode_type/` for the full list with details per
assignment. Never propose reusing a dataset already marked USED there.
