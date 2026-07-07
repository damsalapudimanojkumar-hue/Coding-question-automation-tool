"""Agent 2: interactive problem and dataset design with three HITL phases."""

import json
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_client import call_claude, call_claude_with_tools
from naming import asset_paths, filename_instructions, make_code
from ui.session import TerminalIO
from tools.local_tools import RUN_PYTHON_TOOL, tool_executor
from prompts.problem_dataset_prompt import (
    DATASET_DRAFT_SYSTEM_PROMPT,
    DATASET_EDIT_SYSTEM_PROMPT,
    OPTIONS_SYSTEM_PROMPT,
    REVISION_SYSTEM_PROMPT,
    SYNTHETIC_SYSTEM_PROMPT,
    build_dataset_change_user_prompt,
    build_dataset_draft_user_prompt,
    build_options_user_prompt,
    build_revision_user_prompt,
)

def _between(text, start, end):
    match = re.search(re.escape(start) + r"(.*?)" + re.escape(end), text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_options(text):
    payload = _between(text, "---OPTIONS_JSON---", "---END_OPTIONS_JSON---")
    if not payload:
        match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        payload = match.group(0) if match else ""
    try:
        options = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return []
    required = {"title", "dataset", "learning_objective", "rows", "balance"}
    return [item for item in options if isinstance(item, dict) and required <= item.keys()]


def _parse_design(text):
    plan = _between(text, "---DATASET_PLAN---", "---END_DATASET_PLAN---")
    statement = _between(text, "---PROBLEM_STATEMENT---", "---END_PROBLEM_STATEMENT---")
    return plan, statement


def _question_reference_formats():
    """Load only problem-statement references, excluding evaluation templates."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(root, "knowledge", "reference_formats")
    parts = []
    for name in ("question_json.md", "question_text_patterns.md"):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as file:
                parts.append(file.read())
    return "\n\n---\n\n".join(parts)


def _saved_dataset_summary(workspace, code):
    """Return verified CSV structure for recovery after a missing final response."""
    lines = []
    paths = tuple(asset_paths(workspace, code).values())
    for path in paths:
        if not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            header = next(reader, [])
            row_count = sum(1 for _ in reader)
        lines.append(f"{os.path.relpath(path, workspace)}: {row_count} rows; columns={header}")
    return "\n".join(lines)


def _dataset_snapshot(workspace, code):
    paths = tuple(asset_paths(workspace, code).values())
    return {path: os.stat(path).st_mtime_ns if os.path.isfile(path) else None for path in paths}


def _dataset_was_refreshed(before):
    """Require every output CSV to be created or rewritten during this attempt."""
    return all(
        os.path.isfile(path) and os.stat(path).st_mtime_ns != old_mtime
        for path, old_mtime in before.items()
    )


def _recover_design(state, selected, workspace, code):
    summary = _saved_dataset_summary(workspace, code)
    if not summary:
        return "", ""
    print("Dataset files were saved; recovering the missing final draft without rerunning downloads...")
    prompt = f"""The dataset preparation tool run completed but omitted its final formatted response.
Use the verified file summary and selected option below. Do not claim facts not present here.

SELECTED OPTION: {json.dumps(selected, indent=2)}
VERIFIED FILE SUMMARY:
{summary}

REFERENCE FORMATS:
{_question_reference_formats()}

Return only ---DATASET_PLAN--- and ---PROBLEM_STATEMENT--- blocks in the required format.
The problem statement must end with Important Instructions and contain no em dashes."""
    response = call_claude(system=DATASET_DRAFT_SYSTEM_PROMPT, user=prompt, max_tokens=5000)
    return _parse_design(response)


def _generate_options(state, existing):
    response = call_claude(
        system=OPTIONS_SYSTEM_PROMPT,
        user=build_options_user_prompt(
            state["topic"], state.get("learning_objective", ""),
            state.get("research_output", ""), state.get("wiki_dataset_context", ""),
            json.dumps(existing, indent=2) if existing else None,
        ),
        max_tokens=3000,
    )
    return _parse_options(response)


def _select_option(state, options, io):
    while True:
        if not options:
            io.emit("notice", text="No valid options were returned. Generating another set...")
            options.extend(_generate_options(state, options))
            continue
        io.emit("options", options=options)
        choice = io.ask({"kind": "select_option", "n": len(options)})
        if choice == "M":
            new_options = _generate_options(state, options)
            if new_options:
                options.extend(new_options)
            else:
                io.emit("notice", text="Could not parse more options. Please try again.")
        elif choice == "Q":
            return None
        elif choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        else:
            io.emit("notice", text="Choose an option number, M, or Q.")


def _generate_design(state, selected, workspace, code):
    before = _dataset_snapshot(workspace, code)
    response = call_claude_with_tools(
        system=DATASET_DRAFT_SYSTEM_PROMPT,
        user=build_dataset_draft_user_prompt(
            state["topic"], state.get("learning_objective", ""), selected,
            state.get("research_output", ""), state.get("wiki_instructions_context", ""),
            _question_reference_formats(), workspace, filename_instructions(code),
        ),
        tools=[RUN_PYTHON_TOOL], tool_executor=tool_executor, max_turns=24,
    )
    plan, statement = _parse_design(response)
    if not plan or not statement:
        if _dataset_was_refreshed(before):
            return _recover_design(state, selected, workspace, code)
        return "", ""
    return plan, statement


def _revise_statement(state, statement, io):
    instructions = io.ask({"kind": "text", "prompt": "Describe the problem-statement changes:"})
    response = call_claude(
        system=REVISION_SYSTEM_PROMPT,
        user=build_revision_user_prompt(
            statement, instructions, _question_reference_formats()
        ),
        max_tokens=4000,
    )
    return _between(response, "---PROBLEM_STATEMENT---", "---END_PROBLEM_STATEMENT---")


def _change_dataset(state, selected, plan, statement, workspace, code, io, synthetic=False):
    if synthetic:
        instructions = io.ask({"kind": "text",
                               "prompt": "Synthetic dataset requirements (blank for model defaults):"})
        system = SYNTHETIC_SYSTEM_PROMPT
    else:
        instructions = io.ask({"kind": "text", "prompt": "Describe how to modify the existing dataset:"})
        system = DATASET_EDIT_SYSTEM_PROMPT
    response = call_claude_with_tools(
        system=system,
        user=build_dataset_change_user_prompt(
            selected, statement, plan, instructions or "Use suitable defaults.", workspace,
            _question_reference_formats(), filename_instructions(code),
        ),
        tools=[RUN_PYTHON_TOOL], tool_executor=tool_executor, max_turns=12,
    )
    new_plan, new_statement = _parse_design(response)
    return new_plan or plan, new_statement or statement, instructions


def problem_dataset_agent(state: dict, io=None) -> dict:
    io = io or TerminalIO()
    workspace = state.get("output_dir")
    if not workspace:
        raise ValueError("state['output_dir'] must be initialized before Agent 2 runs")
    code = state.get("assignment_code") or make_code(state["topic"])
    os.makedirs(os.path.join(workspace, "tests"), exist_ok=True)
    io.emit("stage", name="problem_dataset")
    io.emit("log", text="AGENT 2: PROBLEM + DATASET DESIGNER\n"
                        "Phase 1 generates options only. No dataset is downloaded before selection.")

    options = list(state.get("options_table") or state.get("problem_dataset_options") or [])
    if not options:
        options.extend(_generate_options(state, options))

    while True:
        selected = _select_option(state, options, io)
        if selected is None:
            return {
                "options_table": options,
                "problem_dataset_options": options,
                "selected_option": None,
                "approved": False,
                "current_stage": "problem_options_rejected",
            }

        io.emit("log", text="Phase 2: obtaining and inspecting the selected dataset...")
        plan, statement = _generate_design(state, selected, workspace, code)
        if not plan or not statement:
            io.emit("notice", text="[!] Dataset preparation did not complete for this option "
                                   "(source unreachable, or the model ran out of tool turns "
                                   "before saving the CSVs and returning the draft).")
            recovery = io.ask({"kind": "recovery"})
            if recovery == "R":
                plan, statement = _generate_design(state, selected, workspace, code)
            elif recovery == "S":
                plan, statement, _ = _change_dataset(
                    state, selected, "", "", workspace, code, io, synthetic=True
                )
            if not plan or not statement:
                io.emit("notice", text="Still no complete draft. Returning to options.")
                continue

        modification_notes = None
        while True:
            io.emit("draft", plan=plan, statement=statement)
            action = io.ask({"kind": "draft_action"})
            if action == "A":
                return {
                    "options_table": options,
                    "problem_dataset_options": options,
                    "selected_option": selected,
                    "problem_statement_draft": statement,
                    "problem_statement": statement,
                    "dataset_plan": plan,
                    "dataset_files_path": workspace,
                    "output_dir": workspace,
                    "dataset_modification_notes": modification_notes,
                    "approved": True,
                    "current_stage": "problem_dataset_done",
                }
            if action == "EP":
                revised = _revise_statement(state, statement, io)
                if revised:
                    statement = revised
                else:
                    io.emit("notice", text="Revision could not be parsed; keeping the current draft.")
            elif action in {"ED", "SD"}:
                plan, statement, note = _change_dataset(
                    state, selected, plan, statement, workspace, code, io, synthetic=(action == "SD")
                )
                prefix = "Synthetic replacement" if action == "SD" else "Dataset edit"
                modification_notes = f"{prefix}: {note or 'model defaults'}"
            elif action == "B":
                break
            else:
                io.emit("notice", text="Choose A, EP, ED, SD, or B.")
