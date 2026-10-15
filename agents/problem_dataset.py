"""Agent 2: interactive problem and dataset design with three HITL phases."""

import json
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_client import call_claude, call_claude_with_tools
from naming import asset_paths, filename_instructions, images_zip_name, make_code, test_images_zip_relname
from dataset_ops import apply_edits, feature_columns
from ui.session import TerminalIO
from tracing import observe
from tools.image_dataset import describe, validate_image_dataset
from tools.image_sources import build_archives, default_choice, inspect_dataset
from tools.local_tools import RUN_PYTHON_TOOL, tool_executor
from prompts.problem_dataset_prompt import (
    CV_OPTIONS_SYSTEM_PROMPT,
    CV_PROBLEM_STATEMENT_PROMPT,
    DATASET_DRAFT_SYSTEM_PROMPT,
    DATASET_EDIT_SYSTEM_PROMPT,
    OPTIONS_SYSTEM_PROMPT,
    REVISION_SYSTEM_PROMPT,
    SYNTHETIC_SYSTEM_PROMPT,
    build_dataset_change_user_prompt,
    build_dataset_draft_user_prompt,
    build_image_statement_user_prompt,
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


def _is_cv(state):
    return state.get("assignment_type") == "cv"


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
        system=CV_OPTIONS_SYSTEM_PROMPT if _is_cv(state) else OPTIONS_SYSTEM_PROMPT,
        user=build_options_user_prompt(
            state["topic"], state.get("learning_objective", ""),
            state.get("research_output", ""), state.get("wiki_dataset_context", ""),
            json.dumps(existing, indent=2) if existing else None,
            curriculum=state.get("wiki_curriculum", ""),
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


def _generate_design(state, selected, workspace, code, io):
    if _is_cv(state):
        return _generate_image_design(state, selected, workspace, code, io)
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


def _generate_image_design(state, selected, workspace, code, io):
    """CV datasets: download and inspect the real dataset, let the human choose how much of it
    to use, build the two image archives with fixed code, verify them, then draft the problem.
    Returns None when the human goes back to the options."""
    source, source_id = selected.get("source"), selected.get("source_id")
    io.emit("log", text=f"Downloading and inspecting {selected.get('dataset') or source_id} "
                        f"({source}). The first download of a dataset can take a few minutes...")
    try:
        info = inspect_dataset(source, source_id)
    except Exception as exc:  # noqa: BLE001 - network/source failures are reported, not fatal
        io.emit("notice", text=f"[!] Could not use this dataset: {exc}")
        return "", ""

    defaults = default_choice(info, selected.get("classes"))
    choice = io.ask({"kind": "image_sample", "info": info, "defaults": defaults})
    if isinstance(choice, dict) and choice.get("back"):
        return None
    if not isinstance(choice, dict) or len(choice.get("classes") or []) < 2:
        choice = defaults

    io.emit("log", text="Copying the chosen images into the student and hidden test archives...")
    try:
        split_info = build_archives(source, source_id, choice, workspace, code)
    except Exception as exc:  # noqa: BLE001 - e.g. too few readable images in a class
        io.emit("notice", text=f"[!] Could not build the image archives: {exc}")
        return "", ""
    summary, problems, warnings = validate_image_dataset(workspace, code)
    if problems:
        io.emit("notice", text="[!] The image dataset failed verification:\n- " + "\n- ".join(problems))
        return "", ""
    for warning in warnings:
        io.emit("notice", text=warning)

    plan = _image_plan(info, choice, summary, code, split_info)
    io.emit("log", text="Writing the problem statement from the verified dataset...")
    response = call_claude(
        system=CV_PROBLEM_STATEMENT_PROMPT,
        user=build_image_statement_user_prompt(
            state["topic"], state.get("learning_objective", ""), selected,
            state.get("research_output", ""), plan, _question_reference_formats(),
            state.get("wiki_curriculum", ""),
        ),
        max_tokens=5000,
    )
    statement = _between(response, "---PROBLEM_STATEMENT---", "---END_PROBLEM_STATEMENT---")
    return plan, statement


def _image_plan(info, choice, summary, code, split_info):
    """Dataset plan built from verified facts only (no model involved)."""
    amount = ("every image of each class" if choice.get("use_all")
              else f"{choice['images_per_class']} images per class")
    verified = "\n".join(f"- {line}" for line in describe(summary).splitlines())
    hidden = ("hidden images come from the dataset's own official test split" if split_info.get("official_split")
              else f"{round(choice['test_fraction'] * 100)}% of each class held back at random for grading")
    variant = (f"\n\n**Variant used:** `{info['variant_used']}` (the download also has "
               f"{', '.join(v for v in info['variants'] if v != info['variant_used'])})"
               if info.get("variants") else "")
    return (
        f"**Source:** {info['source']} · `{info['source_id']}` ({info['name']})\n\n"
        f"**Full dataset:** {len(info['classes'])} classes, {info['total']:,} images{variant}\n\n"
        f"**Chosen:** {len(choice['classes'])} classes, {amount}; {hidden}; "
        f"duplicate and unreadable images skipped; images keep their original size and format\n\n"
        f"**Files:** `{images_zip_name(code)}` (students get this), "
        f"`{test_images_zip_relname(code)}` (hidden, grading only)\n\n"
        f"**Verified by the pipeline:**\n\n{verified}"
    )


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


def _any_refreshed(before):
    """True if ANY tracked dataset file changed on disk during this attempt."""
    return any(
        os.path.isfile(path) and os.stat(path).st_mtime_ns != old_mtime
        for path, old_mtime in before.items()
    )


def _llm_dataset_op(state, selected, plan, statement, workspace, code, instructions, io, synthetic=False):
    """LLM-driven dataset change (synthetic replacement or freeform edit).
    Reliability fixes vs the old version: ABSOLUTE paths (no cwd ambiguity),
    higher turn budget, and a no-op notice if nothing actually changed."""
    system = SYNTHETIC_SYSTEM_PROMPT if synthetic else DATASET_EDIT_SYSTEM_PROMPT
    before = _dataset_snapshot(workspace, code)
    response = call_claude_with_tools(
        system=system,
        user=build_dataset_change_user_prompt(
            selected, statement, plan, instructions or "Use suitable defaults.", workspace,
            _question_reference_formats(), filename_instructions(code, workspace),
        ),
        tools=[RUN_PYTHON_TOOL], tool_executor=tool_executor, max_turns=24,
    )
    new_plan, new_statement = _parse_design(response)
    if not _any_refreshed(before) and not (new_plan or new_statement):
        io.emit("notice", text="The model reported no changes to the dataset files.")
    return new_plan or plan, new_statement or statement


def _change_dataset(state, selected, plan, statement, workspace, code, io, synthetic=False):
    """Synthetic replacement (SD action): ask requirements, then run the LLM op."""
    prompt = ("Synthetic dataset requirements (blank for model defaults):" if synthetic
              else "Describe how to modify the existing dataset:")
    instructions = io.ask({"kind": "text", "prompt": prompt})
    new_plan, new_statement = _llm_dataset_op(
        state, selected, plan, statement, workspace, code, instructions, io, synthetic=synthetic
    )
    return new_plan, new_statement, instructions


def _refresh_statement_for_facts(statement, new_plan):
    """Update the student-facing statement to match new verified dataset facts
    (used only when preset edits change the feature set)."""
    response = call_claude(
        system=REVISION_SYSTEM_PROMPT,
        user=build_revision_user_prompt(
            statement,
            "The dataset changed. Update any dataset facts (feature list / columns) to match "
            "the verified summary below. Do not invent numbers.\n\n" + new_plan,
            _question_reference_formats(),
        ),
        max_tokens=4000,
    )
    return _between(response, "---PROBLEM_STATEMENT---", "---END_PROBLEM_STATEMENT---")


def _apply_dataset_edits(state, selected, plan, statement, workspace, code, spec, io):
    """Deterministic preset edits (pandas, no LLM) plus optional freeform edit.
    Returns (new_plan, new_statement, note)."""
    ops = {k: spec.get(k) for k in
           ("drop_columns", "rebalance", "resize_factor", "add_noise", "inject_nulls", "impute")}
    try:
        summary = apply_edits(workspace, code, ops)
    except Exception as e:  # noqa: BLE001 - surfaced to the user, never crashes the run
        io.emit("notice", text=f"Preset edit failed: {type(e).__name__}: {e}")
        return plan, statement, "edit failed"

    changes = summary["changes"] or ["no preset changes selected"]
    io.emit("notice", text="Applied: " + "; ".join(changes))
    new_plan = (
        "Dataset (after preset edits)\n"
        f"Train shape: {summary['new_train_shape']} (was {summary['orig_train_shape']})\n"
        f"Test shape: {summary['test_shape']}\n"
        f"Target: {summary['target']}\n"
        f"Features: {summary['features']}\n"
        "Edits applied: " + "; ".join(changes)
    )

    new_statement = statement
    if ops.get("drop_columns"):        # feature set changed -> refresh the statement
        refreshed = _refresh_statement_for_facts(statement, new_plan)
        if refreshed:
            new_statement = refreshed

    freeform = (spec.get("freeform") or "").strip()
    if freeform:                       # extra instructions -> LLM edit on top
        new_plan, new_statement = _llm_dataset_op(
            state, selected, new_plan, new_statement, workspace, code, freeform, io, synthetic=False
        )

    return new_plan, new_statement, "; ".join(changes)


@observe(as_type="agent")
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
        cv = _is_cv(state)
        design = _generate_design(state, selected, workspace, code, io)
        if design is None:                      # Back to options from the image panel
            continue
        plan, statement = design
        if not plan or not statement:
            io.emit("notice", text="[!] Dataset preparation did not complete for this option "
                                   "(see the message above).")
            recovery = io.ask({"kind": "recovery", "assignment_type": state.get("assignment_type")})
            if recovery == "R":
                design = _generate_design(state, selected, workspace, code, io)
                if design is None:
                    continue
                plan, statement = design
            elif recovery == "S" and cv:
                io.emit("notice", text="Synthetic data is not used for image datasets. "
                                       "Pick another option instead.")
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
            action = io.ask({"kind": "draft_action", "assignment_type": state.get("assignment_type")})
            if cv and action in ("SD", "ED"):
                io.emit("notice", text="Synthetic replacement and preset edits only work on "
                                       "tabular data. Use Edit problem or Back to options.")
                continue
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
            elif action == "SD":
                plan, statement, note = _change_dataset(
                    state, selected, plan, statement, workspace, code, io, synthetic=True
                )
                modification_notes = f"Synthetic replacement: {note or 'model defaults'}"
            elif action == "ED":
                spec = io.ask({
                    "kind": "edit_dataset",
                    "assignment_type": state.get("assignment_type", "tabular"),
                    "columns": feature_columns(workspace, code),
                })
                if not isinstance(spec, dict) or spec.get("cancel"):
                    io.emit("notice", text="Edit cancelled; dataset unchanged.")
                else:
                    plan, statement, note = _apply_dataset_edits(
                        state, selected, plan, statement, workspace, code, spec, io
                    )
                    modification_notes = f"Dataset edit: {note}"
            elif action == "B":
                break
            else:
                io.emit("notice", text="Choose A, EP, ED, SD, or B.")
