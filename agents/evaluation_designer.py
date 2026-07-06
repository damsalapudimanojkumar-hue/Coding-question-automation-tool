"""
Agent 3 — Evaluation Designer

Two-phase agent with a human-in-the-loop pause between phases.

PHASE 1: Build reference solution, compute real metric, propose test cases
         → STOPS and shows you the proposal
HITL:    You review in terminal, type approve / edit / regenerate
PHASE 2: Generate conftest.py, test_solution.py, solution.ipynb,
         question.json, pytest.ini, requirements.txt

Input from state:
    state["problem_statement"]     <- from Agent 2
    state["dataset_plan"]          <- from Agent 2
    state["dataset_files_path"]    <- where CSVs live (from Agent 2)
    state["wiki_skill_context"]    <- from Agent 0

Output to state:
    state["approved_test_cases"]   <- what was approved after HITL
    state["generated_files_path"]  <- where files were saved
    state["evaluation_complete"]   <- True when done
"""

import sys
import os
import re
import json
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_client import call_claude_with_tools
from naming import filename_instructions, asset_relnames, make_code
from tools.local_tools import ALL_TOOLS, tool_executor
from prompts.evaluation_prompt import (
    EVALUATION_PHASE1_PROMPT,
    EVALUATION_PHASE2_PROMPT,
    build_phase1_user_prompt,
    build_phase2_user_prompt,
)

def _parse_test_proposal(response_text: str) -> str:
    """Extract the test case proposal block from Phase 1 response."""
    match = re.search(
        r"---TEST_CASE_PROPOSAL---(.*?)---END_TEST_CASE_PROPOSAL---",
        response_text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return response_text


def _extract_solution_code(response_text: str) -> str:
    """
    Try to pull the reference solution code from Phase 1 output.
    Looks for the last ```python ... ``` block in the response,
    which is typically the full solution code the agent ran.
    """
    blocks = re.findall(r"```python(.*?)```", response_text, re.DOTALL)
    if blocks:
        # Return the longest code block (most likely the full solution)
        return max(blocks, key=len).strip()
    return ""


def _hitl_review(proposal: str) -> tuple:
    """
    Show the proposed test cases and handle review with three modes:
    - Approve: use as-is
    - Discuss: interactive loop — user gives feedback (vague or specific), 
               agent proposes revised test cases, repeat until user accepts
    - Regenerate: redo Phase 1 from scratch with feedback
    """
    print("\n" + "="*60)
    print("⏸️   CHECKPOINT — REVIEW PROPOSED TEST CASES")
    print("="*60)
    print(proposal)
    print("\n" + "-"*60)
    print("Options:")
    print("  [A] Approve — generate files with these test cases")
    print("  [D] Discuss — give feedback, I'll propose revisions (iterative)")
    print("  [R] Regenerate — redo Phase 1 from scratch with feedback")
    print("-"*60)

    while True:
        choice = input("\nYour choice [A/D/R]: ").strip().upper()

        if choice == "A":
            print("\n✅ Approved. Proceeding to Phase 2 — generating files...\n")
            return "approve", proposal

        elif choice == "D":
            return _discuss_and_refine(proposal)

        elif choice == "R":
            feedback = input("\nWhat should change? (brief description): ").strip()
            print("\n🔄 Regenerating Phase 1 with your feedback...\n")
            return "regenerate", feedback

        else:
            print("Please type A, D, or R.")


def _discuss_and_refine(current_proposal: str) -> tuple:
    """
    Interactive refinement loop.
    User gives feedback → agent proposes revised test cases → repeat until accepted.
    """
    print("\n" + "="*60)
    print("💬  DISCUSS & REFINE MODE")
    print("="*60)
    print("Give feedback in any form (vague or specific).")
    print("I'll propose a revised test case list each round.")
    print("Type 'accept' to finalize, 'back' to return to main menu.")
    print("="*60 + "\n")

    proposal = current_proposal

    while True:
        feedback = input("\nYour feedback (or 'accept' / 'back'): ").strip()

        if feedback.lower() == "accept":
            print("\n✅ Accepted. Proceeding to Phase 2...\n")
            return "approve", proposal

        if feedback.lower() == "back":
            print("\n↩️  Returning to main menu...\n")
            return _hitl_review(current_proposal)  # back to A/D/R menu

        # Ask the model to revise the proposal based on feedback
        print("\n🤔  Revising test cases based on your feedback...")
        revised = _call_model_for_revision(proposal, feedback)

        if not revised:
            print("⚠️  Could not parse revised proposal. Try again or type 'back'.")
            continue

        print("\n" + "="*60)
        print("📝  REVISED TEST CASE PROPOSAL")
        print("="*60)
        print(revised)
        print("="*60 + "\n")

        proposal = revised  # update for next iteration


def _call_model_for_revision(current_proposal: str, feedback: str) -> str:
    """
    Call the model (without tools) to revise the test case proposal.
    Returns the revised proposal text, or empty string on failure.
    """
    from claude_client import call_claude
    from prompts.evaluation_prompt import EVALUATION_PHASE1_PROMPT

    revision_prompt = f"""You are revising a test case proposal for an ML assignment evaluation.

CURRENT PROPOSAL:
{current_proposal}

HUMAN FEEDBACK:
{feedback}

INSTRUCTIONS:
- Revise the test case list based on the feedback
- Keep the same output format (PY1..PYN with weights, detection notes)
- Total weight must still sum to 100
- Be concise — only output the revised proposal block (between ---TEST_CASE_PROPOSAL--- and ---END_TEST_CASE_PROPOSAL---)
- Do NOT add extra commentary"""

    try:
        response = call_claude(
            system=EVALUATION_PHASE1_PROMPT,
            user=revision_prompt,
            max_tokens=3000,
        )
        # Extract just the proposal block
        import re
        match = re.search(
            r"---TEST_CASE_PROPOSAL---(.*?)---END_TEST_CASE_PROPOSAL---",
            response,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        # Fallback: return whole response if delimiters missing
        return response.strip()
    except Exception as e:
        print(f"[Revision call failed: {e}]")
        return ""


def _finalize_question_json(workspace: str) -> str:
    """
    Post-process the generated question file: overwrite question_id and
    ide_session_id with real library-generated UUIDs (the model is unreliable
    at producing valid UUIDs), then save the file named by its question_id
    (<uuid>.json) and remove the generic question.json.

    Returns the new filename, or "" if no question file was found.
    """
    src = os.path.join(workspace, "question.json")
    if not os.path.isfile(src):
        return ""

    try:
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"   [WARN] could not parse question.json ({e}); leaving as-is.")
        return "question.json"

    question_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    # Structure is a JSON array with one question object.
    records = data if isinstance(data, list) else [data]
    for rec in records:
        if isinstance(rec, dict):
            rec["question_id"] = question_id
            rec["ide_session_id"] = session_id

    dst = os.path.join(workspace, f"{question_id}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(records if isinstance(data, list) else records[0], f, indent=2, ensure_ascii=False)
    os.remove(src)
    print(f"   question.json -> {question_id}.json (question_id + ide_session_id set)")
    return f"{question_id}.json"


def evaluation_designer_agent(state: dict) -> dict:
    """LangGraph node function for Agent 3."""

    topic = state["topic"]
    problem_statement = state.get("problem_statement", "")
    dataset_plan = state.get("dataset_plan", "")
    workspace = state.get("output_dir") or state.get("dataset_files_path")
    if not workspace:
        raise ValueError("state['output_dir'] must be initialized before Agent 3 runs")
    wiki_skill_context = state.get("wiki_skill_context", "")
    wiki_reference_formats = state.get("wiki_reference_formats", "")
    wiki_examples = state.get("wiki_examples", "")
    code = state.get("assignment_code") or make_code(topic)
    filenames = filename_instructions(code)

    os.makedirs(workspace, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🧪  AGENT 3: EVALUATION DESIGNER")
    print(f"{'='*60}")
    print(f"Topic: {topic}")
    print(f"Workspace: {workspace}")

    # ── PHASE 1 LOOP (re-runs if human chooses Regenerate) ─────────────
    extra_feedback = ""
    phase1_response = ""
    approved_test_cases = ""
    reference_solution_code = ""

    while True:
        print("\n📐 Phase 1: Building reference solution + proposing test cases...")

        phase1_prompt = build_phase1_user_prompt(
            topic=topic,
            problem_statement=problem_statement,
            dataset_plan=dataset_plan,
            workspace=workspace,
            wiki_skill_context=wiki_skill_context,
            filenames=filenames,
        )

        if extra_feedback:
            phase1_prompt += (
                f"\n\nPREVIOUS PROPOSAL WAS REJECTED. Human feedback:\n"
                f"{extra_feedback}\n"
                f"Please revise your test case proposal accordingly."
            )

        phase1_response = call_claude_with_tools(
            system=EVALUATION_PHASE1_PROMPT,
            user=phase1_prompt,
            tools=ALL_TOOLS,
            tool_executor=tool_executor,
            max_turns=15,
        )

        proposal = _parse_test_proposal(phase1_response)
        reference_solution_code = _extract_solution_code(phase1_response)

        # ── HITL pause ──────────────────────────────────────────────────
        action, feedback_or_cases = _hitl_review(proposal)

        if action == "approve":
            approved_test_cases = feedback_or_cases
            break
        else:
            # action == "regenerate"
            extra_feedback = feedback_or_cases
            continue

    # ── PHASE 2: Generate all test files ───────────────────────────────
    print("\n📁 Phase 2: Generating test files...")

    phase2_prompt = build_phase2_user_prompt(
        topic=topic,
        problem_statement=problem_statement,
        dataset_plan=dataset_plan,
        workspace=workspace,
        wiki_skill_context=wiki_skill_context,
        wiki_reference_formats=wiki_reference_formats,
        approved_test_cases=approved_test_cases,
        reference_solution_code=reference_solution_code,
        wiki_examples=wiki_examples,
        filenames=filenames,
    )

    phase2_response = call_claude_with_tools(
        system=EVALUATION_PHASE2_PROMPT,
        user=phase2_prompt,
        tools=ALL_TOOLS,
        tool_executor=tool_executor,
        max_turns=20,
    )

    print("\n" + "="*60)
    print("AGENT 3 PHASE 2 RESPONSE:")
    print("="*60)
    print(phase2_response)
    print("="*60)

    # Generate real UUIDs and rename question.json -> <question_id>.json
    question_file = _finalize_question_json(workspace)

    # Confirm which files landed on disk. conftest.py lives only in tests/ now
    # (no duplicate root copy).
    gt_name = os.path.basename(asset_relnames(code)["ground_truth"])
    expected_files = [
        "solution.ipynb",
        os.path.join("tests", "solution_variants.ipynb"),
        os.path.join("tests", "__init__.py"),
        os.path.join("tests", "conftest.py"),
        os.path.join("tests", "test_solution.py"),
        os.path.join("tests", gt_name),
        question_file or "question.json",
        "pytest.ini",
        "requirements.txt",
    ]

    print(f"\n📁 Files in {workspace}:")
    for fname in expected_files:
        fpath = os.path.join(workspace, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"   ✅ {fname} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ {fname} (missing)")

    return {
        "approved_test_cases": approved_test_cases,
        "generated_files_path": workspace,
        "question_file": question_file,
        "evaluation_complete": True,
    }
