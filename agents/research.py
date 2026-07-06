"""
Agent 1 — Research

Input it reads from state:
    state["topic"]                 -> e.g. "Logistic Regression"
    state["learning_objective"]    -> e.g. "Binary classification with
                                       probability outputs and decision
                                       boundary interpretation"
    state["wiki_research_context"] -> filled by Agent 0 (Wiki Loader),
                                       must run BEFORE this agent

Output it writes back into state:
    state["research_output"]       -> the structured research brief (text)

This agent does NOT decide the final problem statement or dataset.
It only researches and reports back. Agent 2 (Problem + Dataset Designer)
consumes research_output next.
"""

import sys
import os

# Allow running this file standalone via test_research.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_client import call_claude_with_search
from prompts.research_prompt import RESEARCH_SYSTEM_PROMPT, build_research_user_prompt


def research_agent(state: dict) -> dict:
    """
    LangGraph node function. Receives the full pipeline state dict,
    returns only the fields this node adds/updates.
    """
    topic = state["topic"]
    learning_objective = state.get("learning_objective", "")
    wiki_research_context = state.get("wiki_research_context", "")

    if not wiki_research_context:
        # Defensive: Agent 0 should always run first and populate this.
        # If it's empty, research still works but loses the "avoid
        # repeating past datasets" capability — worth surfacing loudly.
        print("  [WARNING] wiki_research_context is empty — Wiki Loader "
              "may not have run, or no past assignments exist yet for "
              "this type. Research will proceed without internal memory.")

    user_prompt = build_research_user_prompt(
        topic=topic,
        learning_objective=learning_objective,
        wiki_research_context=wiki_research_context,
    )

    print(f"\n{'='*60}")
    print(f"🔍  AGENT 1: RESEARCH")
    print(f"{'='*60}")
    print(f"Researching: {topic}")
    print("Searching the web and cross-referencing internal wiki...\n")

    response_text = call_claude_with_search(
        system=RESEARCH_SYSTEM_PROMPT,
        user=user_prompt,
    )

    print(response_text)
    print(f"\n{'='*60}\n")

    return {"research_output": response_text}