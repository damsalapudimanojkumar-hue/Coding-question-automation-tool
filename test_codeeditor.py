"""
test_codeeditor.py — CLI driver for the code-editor (function-based) pipeline.

Two modes:

  1. Deterministic wiring check (NO LLM, NO cost):
         python test_codeeditor.py --from-config knowledge/examples/code_editor_type/bce_config.json
     Loads a ready config, runs only Agent 0 (loader) + Agent 3 (generate),
     and prints the computed outputs + the deliverable zip path. Use this to
     verify the plumbing end-to-end without spending credits.

  2. Full design run (LLM, costs credits):
         python test_codeeditor.py --topic "Softmax" \
             --objective "Implement a numerically stable softmax" [--research]
     Runs Agent 0 -> (optional Research) -> Agent 2 design (with HITL review)
     -> Agent 3 generate.
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows consoles choke on the emoji banners the agents print.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agents.wiki_loader import wiki_loader_agent
from agents.codeeditor_design import codeeditor_design_agent
from agents.codeeditor_generate import codeeditor_generate_agent
from ui.session import TerminalIO
from claude_client import reset_cost, cost_summary


def main():
    parser = argparse.ArgumentParser(description="Code-editor question pipeline (CLI)")
    parser.add_argument("--topic", default="Binary Cross-Entropy Loss")
    parser.add_argument("--objective", default="Implement a numerically stable BCE loss from scratch")
    parser.add_argument("--code", default=None, help="short assignment code (e.g. BCE)")
    parser.add_argument("--from-config", default=None,
                        help="skip the LLM: load this config JSON and just generate")
    parser.add_argument("--research", action="store_true",
                        help="run the research agent for inspiration (costs credits)")
    parser.add_argument("--difficulty", default="", choices=["", "EASY", "MEDIUM", "HARD"],
                        help="target difficulty (default: auto)")
    parser.add_argument("--num-tests", type=int, default=10, help="target number of test cases")
    parser.add_argument("--candidates", type=int, default=1,
                        help="how many problem candidates to offer (>1 enables the picker)")
    parser.add_argument("--no-examples", action="store_true",
                        help="do NOT append an Examples section to the description")
    args = parser.parse_args()

    io = TerminalIO()
    reset_cost()

    state = {
        "topic": args.topic,
        "learning_objective": args.objective,
        "assignment_type": "tabular",        # unused by code-editor, kept for state parity
        "config_type": "code_editor_type",
        "assignment_code": args.code,
        "codeeditor_difficulty": args.difficulty,
        "codeeditor_num_tests": args.num_tests,
        "codeeditor_num_candidates": args.candidates,
        "codeeditor_include_examples": not args.no_examples,
    }

    # Agent 0 — Wiki Loader (routes to the code-editor branch)
    state.update(wiki_loader_agent(state))

    if args.from_config:
        with open(args.from_config, "r", encoding="utf-8") as f:
            state["codeeditor_config"] = json.load(f)
        print(f"\n[from-config] Loaded {args.from_config} — skipping design agent.")
    else:
        if args.research:
            from agents.research import research_agent
            state.update(research_agent(state))
        # Agent 2 — Design (LLM + HITL review)
        state.update(codeeditor_design_agent(state, io))

    # Agent 3 — Generate (deterministic)
    state.update(codeeditor_generate_agent(state, io))

    cost = cost_summary()
    print("\n" + "=" * 60)
    print(f"Deliverable : {state.get('codeeditor_deliverable')}")
    print(f"Run cost    : ${cost['cost_usd']:.4f}  "
          f"({cost['total_tokens']} tokens, {cost['calls']} calls)")
    print("=" * 60)


if __name__ == "__main__":
    main()
