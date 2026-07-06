"""Agent 2: interactive problem and dataset design with three HITL phases."""

import json
import csv
import os
import re
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_client import call_claude, call_claude_with_tools
from naming import asset_paths, filename_instructions, make_code
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


def _clip(value, width):
    value = " ".join(str(value or "").split())
    return textwrap.shorten(value, width=width, placeholder="...").ljust(width)


def _print_options(options):
    widths = (3, 34, 25, 34, 9, 10)
    headers = ("#", "Problem Title", "Dataset", "Learning Objective", "Rows", "Balance")
    rule = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print("\n" + rule)
    print("| " + " | ".join(_clip(v, w) for v, w in zip(headers, widths)) + " |")
    print(rule)
    for index, option in enumerate(options, 1):
        row = (index, option.get("title"), option.get("dataset"),
               option.get("learning_objective"), option.get("rows"), option.get("balance"))
        print("| " + " | ".join(_clip(v, w) for v, w in zip(row, widths)) + " |")
    print(rule)
    print("\n[1-{}] Select   [M] More options   [Q] Change topic".format(len(options)))


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


def _select_option(state, options):
    while True:
        if not options:
            print("No valid options were returned. Generating another set...")
            options.extend(_generate_options(state, options))
            continue
        _print_options(options)
        choice = input("\nYour choice: ").strip().upper()
        if choice == "M":
            new_options = _generate_options(state, options)
            if new_options:
                options.extend(new_options)
            else:
                print("Could not parse more options. Please try again.")
        elif choice == "Q":
            return None
        elif choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        else:
            print("Choose an option number, M, or Q.")


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


def _show_draft(plan, statement):
    print("\n" + "=" * 60)
    print("DATASET LOADED AND PREPARED")
    print("=" * 60)
    print(plan or "[Dataset plan was not returned]")
    print("\n" + "-" * 60)
    print("PROBLEM STATEMENT DRAFT:")
    print("-" * 60)
    print(statement or "[Problem statement was not returned]")
    print("-" * 60)
    print("[A] Approve  [EP] Edit problem  [ED] Edit dataset")
    print("[SD] Synthetic dataset  [B] Back to options")


def _revise_statement(state, statement):
    instructions = input("Describe the problem-statement changes: ").strip()
    response = call_claude(
        system=REVISION_SYSTEM_PROMPT,
        user=build_revision_user_prompt(
            statement, instructions, _question_reference_formats()
        ),
        max_tokens=4000,
    )
    return _between(response, "---PROBLEM_STATEMENT---", "---END_PROBLEM_STATEMENT---")


def _change_dataset(state, selected, plan, statement, workspace, code, synthetic=False):
    if synthetic:
        instructions = input("Synthetic dataset requirements (Enter for model defaults): ").strip()
        system = SYNTHETIC_SYSTEM_PROMPT
    else:
        instructions = input("Describe how to modify the existing dataset: ").strip()
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


def problem_dataset_agent(state: dict) -> dict:
    workspace = state.get("output_dir")
    if not workspace:
        raise ValueError("state['output_dir'] must be initialized before Agent 2 runs")
    code = state.get("assignment_code") or make_code(state["topic"])
    os.makedirs(os.path.join(workspace, "tests"), exist_ok=True)
    print("\n" + "=" * 60)
    print("AGENT 2: PROBLEM + DATASET DESIGNER")
    print("=" * 60)
    print("Phase 1 generates options only. No dataset is downloaded before selection.")

    options = list(state.get("options_table") or state.get("problem_dataset_options") or [])
    if not options:
        options.extend(_generate_options(state, options))

    while True:
        selected = _select_option(state, options)
        if selected is None:
            return {
                "options_table": options,
                "problem_dataset_options": options,
                "selected_option": None,
                "approved": False,
                "current_stage": "problem_options_rejected",
            }

        print("\nPhase 2: obtaining and inspecting the selected dataset...")
        plan, statement = _generate_design(state, selected, workspace, code)
        if not plan or not statement:
            print("\n[!] Dataset preparation did not complete for this option.")
            print("    (The dataset was likely unreachable, or the model ran out of tool")
            print("     turns before it saved the CSVs and returned the draft.)")
            recovery = input("Choose: [R] retry same option  [S] synthetic dataset instead  "
                             "[B] back to options: ").strip().upper()
            if recovery == "R":
                plan, statement = _generate_design(state, selected, workspace, code)
            elif recovery == "S":
                plan, statement, _ = _change_dataset(
                    state, selected, "", "", workspace, code, synthetic=True
                )
            if not plan or not statement:
                print("Still no complete draft. Returning to options.\n")
                continue

        modification_notes = None
        while True:
            _show_draft(plan, statement)
            action = input("\nYour choice [A/EP/ED/SD/B]: ").strip().upper()
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
                revised = _revise_statement(state, statement)
                if revised:
                    statement = revised
                else:
                    print("Revision could not be parsed; keeping the current draft.")
            elif action in {"ED", "SD"}:
                plan, statement, note = _change_dataset(
                    state, selected, plan, statement, workspace, code, synthetic=(action == "SD")
                )
                prefix = "Synthetic replacement" if action == "SD" else "Dataset edit"
                modification_notes = f"{prefix}: {note or 'model defaults'}"
            elif action == "B":
                break
            else:
                print("Choose A, EP, ED, SD, or B.")
