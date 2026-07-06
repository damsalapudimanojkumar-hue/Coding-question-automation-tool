"""
test_research.py

Standalone test for Agent 1 (Research). Runs Wiki Loader first (to get
real internal context), then runs Research against it with a real topic
and a real Claude API call + web search.

Run with:
    python test_research.py

Requires your OPENROUTER_API_KEY to be set in .env (see SETUP.md).
"""

from agents.wiki_loader import wiki_loader_agent
from agents.research import research_agent

# ---- Change this to test different topics ----
TOPIC = "Logistic Regression"
LEARNING_OBJECTIVE = (
    "Build a binary classifier using LogisticRegression, interpret "
    "predicted probabilities (not just hard class labels), and understand "
    "the decision boundary / threshold tuning tradeoff."
)
ASSIGNMENT_TYPE = "tabular"
CONFIG_TYPE = "vscode_type"
# ------------------------------------------------

if __name__ == "__main__":
    print(f"Testing Research agent on topic: '{TOPIC}'\n")

    # Step 1: run Wiki Loader first, same as the real pipeline would.
    state = {
        "topic": TOPIC,
        "learning_objective": LEARNING_OBJECTIVE,
        "assignment_type": ASSIGNMENT_TYPE,
        "config_type": CONFIG_TYPE,
    }
    wiki_output = wiki_loader_agent(state)
    state.update(wiki_output)

    # Step 2: run Research using the wiki context just loaded.
    research_output = research_agent(state)
    state.update(research_output)

    print("\n" + "=" * 60)
    print("Done. research_output is now in state, ready for Agent 2.")
    print("Paste the full output above back to continue.")
    print("=" * 60)