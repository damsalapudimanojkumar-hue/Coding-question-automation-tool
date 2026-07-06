"""
test_wiki_loader.py

STEP 2 — run this to confirm the Wiki Loader correctly reads and filters
the knowledge/ folder. No API key needed for this step (pure file I/O).

Usage:
    python test_wiki_loader.py
"""

from agents.wiki_loader import wiki_loader_agent


def main():
    print("Testing Wiki Loader with assignment_type='tabular', config_type='vscode_type'...")

    state = {
        "assignment_type": "tabular",
        "config_type": "vscode_type",
    }

    result = wiki_loader_agent(state)

    print("\n" + "─" * 60)
    print("RESEARCH CONTEXT (first 600 chars):")
    print("─" * 60)
    print(result["wiki_research_context"][:600])

    print("\n" + "─" * 60)
    print("SKILL CONTEXT (first 400 chars):")
    print("─" * 60)
    print(result["wiki_skill_context"][:400])

    print("\n" + "─" * 60)
    print("INSTRUCTIONS CONTEXT (first 400 chars):")
    print("─" * 60)
    print(result["wiki_instructions_context"][:400])

    print("\n" + "─" * 60)
    print("DATASET CONTEXT (first 400 chars):")
    print("─" * 60)
    print(result["wiki_dataset_context"][:400])

    print("\n\nIf all four sections above show real content (not errors),")
    print("the Wiki Loader is working. Paste this output back to continue.")


if __name__ == "__main__":
    main()
