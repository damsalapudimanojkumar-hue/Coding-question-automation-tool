"""
test_problem_dataset.py

Standalone test for Agent 2 (Problem + Dataset Designer).
Chains: Wiki Loader -> Research -> Problem + Dataset Designer

Run with:
    python test_problem_dataset.py

What to expect:
  - Wiki Loader runs (no API call, instant)
  - Research runs (web search, ~30 seconds)
  - Agent 2 runs (code execution loop, may take 1-3 minutes while it
    fetches/inspects/cleans the dataset and writes the problem statement)
  - At the end you should see a DATASET_PLAN and PROBLEM_STATEMENT printed
  - Three CSV files saved to /home/claude/workspace/ (or C:/... on Windows)

Paste the full terminal output back to continue to the human checkpoint.
"""

import os
from agents.wiki_loader import wiki_loader_agent
from agents.research import research_agent
from agents.problem_dataset import problem_dataset_agent

# ── Change these to test different topics ──────────────────────────────
TOPIC = "Logistic Regression"
LEARNING_OBJECTIVE = (
    "Build a binary classifier using logistic regression, interpret "
    "predicted probabilities (not just hard class labels), and understand "
    "the decision boundary and threshold tuning tradeoff."
)
ASSIGNMENT_TYPE = "tabular"
CONFIG_TYPE = "vscode_type"
ASSIGNMENT_CODE = "LOGREG"      # suffixes data/ground_truth files
# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Building assignment for: '{TOPIC}'")
    print("This will take 1-3 minutes (fetching + inspecting real dataset).\n")

    state = {
        "topic": TOPIC,
        "learning_objective": LEARNING_OBJECTIVE,
        "assignment_type": ASSIGNMENT_TYPE,
        "config_type": CONFIG_TYPE,
        "assignment_code": ASSIGNMENT_CODE,
    }

    # Step 1: Wiki Loader
    state.update(wiki_loader_agent(state))

    # Step 2: Research (uses web search)
    state.update(research_agent(state))

    # Step 3: Problem + Dataset Designer (uses code execution)
    state.update(problem_dataset_agent(state))

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    print("\n📋 DATASET PLAN:")
    print(state.get("dataset_plan", "[not set]"))

    print("\n📝 PROBLEM STATEMENT:")
    print(state.get("problem_statement", "[not set]"))

    print("\n📁 FILES SAVED TO:")
    workspace = state.get("dataset_files_path", "")
    print(f"   {workspace}")
    from naming import asset_paths
    for fname, fpath in asset_paths(workspace, state.get("assignment_code", "")).items():
        if os.path.isfile(fpath):
            print(f"   ✅ {os.path.relpath(fpath, workspace)}")
        else:
            print(f"   ❌ {os.path.relpath(fpath, workspace)} (missing)")

    print("\n" + "=" * 60)
    print("Paste this full output back to continue to Checkpoint 1.")
    print("=" * 60)
