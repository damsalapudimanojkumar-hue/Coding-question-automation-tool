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
from ui.session import TerminalIO
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


def _hitl_review(proposal: str, io) -> tuple:
    """
    Show the proposed test cases and handle review with three modes:
    - Approve: use as-is
    - Discuss: iterative feedback loop that proposes revised test cases
    - Regenerate: redo Phase 1 from scratch with feedback
    """
    io.emit("test_proposal", text=proposal)

    while True:
        choice = io.ask({"kind": "eval_action"})

        if choice == "A":
            io.emit("log", text="Approved. Generating files...")
            return "approve", proposal

        elif choice == "D":
            return _discuss_and_refine(proposal, io)

        elif choice == "R":
            feedback = io.ask({"kind": "text",
                               "prompt": "What should change? (brief description):"})
            io.emit("log", text="Regenerating test-case proposal with your feedback...")
            return "regenerate", feedback

        else:
            io.emit("notice", text="Please choose Approve, Discuss, or Regenerate.")


def _discuss_and_refine(current_proposal: str, io) -> tuple:
    """
    Interactive refinement loop.
    User gives feedback -> agent proposes revised test cases -> repeat until accepted.
    """
    proposal = current_proposal

    while True:
        feedback = io.ask({"kind": "text",
                           "prompt": "Feedback to refine the test cases "
                                     "(or type 'accept' to finalize, 'back' for the menu):"})

        if feedback.lower() == "accept":
            io.emit("log", text="Accepted. Generating files...")
            return "approve", proposal

        if feedback.lower() == "back":
            return _hitl_review(current_proposal, io)  # back to A/D/R menu

        io.emit("log", text="Revising test cases based on your feedback...")
        revised = _call_model_for_revision(proposal, feedback)

        if not revised:
            io.emit("notice", text="Could not parse a revised proposal. Try again or type 'back'.")
            continue

        io.emit("test_proposal", text=revised)
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


def evaluation_designer_agent(state: dict, io=None) -> dict:
    """LangGraph node function for Agent 3."""
    io = io or TerminalIO()

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

    io.emit("stage", name="evaluation")
    io.emit("log", text=f"AGENT 3: EVALUATION DESIGNER — {topic}")

    # ── PHASE 1 LOOP (re-runs if human chooses Regenerate) ─────────────
    extra_feedback = ""
    phase1_response = ""
    approved_test_cases = ""
    reference_solution_code = ""

    while True:
        io.emit("log", text="Phase 1: building reference solution + proposing test cases "
                            "(this runs code, can take a minute)...")

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
        action, feedback_or_cases = _hitl_review(proposal, io)

        if action == "approve":
            approved_test_cases = feedback_or_cases
            break
        else:
            # action == "regenerate"
            extra_feedback = feedback_or_cases
            continue

    # ── PHASE 2: Generate all test files ───────────────────────────────
    io.emit("log", text="Phase 2: generating test files...")

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

    call_claude_with_tools(
        system=EVALUATION_PHASE2_PROMPT,
        user=phase2_prompt,
        tools=ALL_TOOLS,
        tool_executor=tool_executor,
        max_turns=20,
    )

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

    file_report = []
    for fname in expected_files:
        fpath = os.path.join(workspace, fname)
        exists = os.path.isfile(fpath)
        size_kb = round(os.path.getsize(fpath) / 1024, 1) if exists else 0.0
        file_report.append({"name": fname, "exists": exists, "size_kb": size_kb})
        print(f"   {'OK' if exists else 'MISSING'} {fname} ({size_kb} KB)")

    io.emit("files", files=file_report, workspace=workspace, question_file=question_file)

    return {
        "approved_test_cases": approved_test_cases,
        "generated_files_path": workspace,
        "question_file": question_file,
        "generated_files": file_report,
        "evaluation_complete": True,
    }
