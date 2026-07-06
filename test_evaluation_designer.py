"""
test_evaluation_designer.py

Chains: Wiki Loader -> Research -> Problem+Dataset -> Evaluation Designer

The pipeline will PAUSE at the test case review checkpoint and wait
for your input in the terminal before generating files.

Run with:
    python test_evaluation_designer.py

Expected flow:
  1. Wiki Loader   (~2 seconds, no API call)
  2. Research      (~30 seconds, web search)
  3. Agent 2       (options HITL, then dataset fetch + clean + approval)
  4. Agent 3 Phase 1  (~1 minute, builds solution, computes real metric)
  5. *** TERMINAL PAUSE — you review and type A/E/R ***
  6. Agent 3 Phase 2  (~2 minutes, generates all test files)

All output files land in an isolated topic folder such as:
  outputs/bagging/

including dataset CSVs, solution.ipynb, conftest.py,
test_solution.py, question.json, pytest.ini, requirements.txt.
"""

import os
from agents.wiki_loader import wiki_loader_agent
from agents.research import research_agent
from agents.problem_dataset import problem_dataset_agent
from agents.evaluation_designer import evaluation_designer_agent

TOPIC = "Boosting"

LEARNING_OBJECTIVE = (
    "Apply Boosting to improve predictive performance by sequentially "
    "training weak learners, understand how each learner focuses on "
    "previous errors, compare Boosting performance with a single model "
    "and Bagging, interpret ensemble predictions, and evaluate performance "
    "using appropriate classification or regression metrics."
)

ASSIGNMENT_TYPE = "tabular"

CONFIG_TYPE = "vscode_type"     # IDE-based coding assignment

ASSIGNMENT_CODE = "BST"         # suffixes datasets/ground_truth/zips (dataset_train_BAG.csv, ...)


if __name__ == "__main__":
    print(f"Building full assignment for: '{TOPIC}'")
    print("This will take 5-8 minutes total.\n")
    print("You will be asked to review test cases partway through.\n")

    state = {
        "topic": TOPIC,
        "learning_objective": LEARNING_OBJECTIVE,
        "assignment_type": ASSIGNMENT_TYPE,
        "config_type": CONFIG_TYPE,
        "assignment_code": ASSIGNMENT_CODE,
    }

    state.update(wiki_loader_agent(state))
    state.update(research_agent(state))
    state.update(problem_dataset_agent(state))
    # evaluation_designer_agent reads wiki_reference_formats from state and
    # injects it into its Phase 2 question.json generation prompt.
    state.update(evaluation_designer_agent(state))

    # ── Final summary ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

    output_dir = state.get("generated_files_path", state["output_dir"])
    print(f"\nAll files saved to: {output_dir}")
    print("\nTo run local tests:")
    print(f"  cd {output_dir}")
    print(f"  pytest -vv")

    print("\n" + "="*60)
