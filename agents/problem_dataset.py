"""Agent 2: interactive problem and dataset design with three HITL phases."""

import json
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_client import call_claude, call_claude_with_tools
from naming import asset_paths, filename_instructions, make_code
from dataset_ops import apply_edits, feature_columns
from ui.session import TerminalIO
from tracing import observe
from tools.dataset_verify import verify_options
from tools.image_dataset import describe, validate_image_dataset
from tools.image_layout import describe_layout
from tools.image_sources import build_archives, default_choice, inspect_dataset, source_by_key
from tools.local_tools import RUN_PYTHON_TOOL, tool_executor
from prompts.problem_dataset_prompt import (
    CV_OPTIONS_SYSTEM_PROMPT,
    CV_PROBLEM_STATEMENT_PROMPT,
    DATA_REQUIREMENT_PROMPT,
    DATASET_DRAFT_SYSTEM_PROMPT,
    DATASET_EDIT_SYSTEM_PROMPT,
    OPTIONS_SYSTEM_PROMPT,
    REVISION_SYSTEM_PROMPT,
    SYNTHETIC_SYSTEM_PROMPT,
    build_dataset_change_user_prompt,
    build_dataset_draft_user_prompt,
    build_image_statement_user_prompt,
    build_options_user_prompt,
    build_requirement_user_prompt,
    build_revision_user_prompt,
)

# Used only when the objective could not be analysed: the safest assumption for a CV assignment.
DEFAULT_REQUIREMENT = {"labels": "required", "grading": "accuracy", "task": "image classification",
                       "needs": "labelled images in a few visually distinct classes",
                       "why": "default, because the learning objective could not be analysed"}
MIN_VERIFIED_OPTIONS = 3
OPTION_ROUNDS = 3

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


def _parse_requirement(text):
    payload = _between(text, "---REQUIREMENT_JSON---", "---END_REQUIREMENT_JSON---")
    if not payload:
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        payload = match.group(0) if match else ""
    try:
        req = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(req, dict):
        return None
    labels, grading = str(req.get("labels", "")).lower(), str(req.get("grading", "")).lower()
    if labels not in ("required", "optional", "none") or grading not in ("accuracy", "structural"):
        return None
    if grading == "accuracy":
        labels = "required"            # accuracy can only be measured against labels
    return {"labels": labels, "grading": grading, "task": str(req.get("task", "")),
            "needs": str(req.get("needs", "")), "why": str(req.get("why", ""))}


def _data_requirement(state, io):
    """Write down what the learning objective needs from a dataset BEFORE any dataset is
    suggested; every later step (suggesting, checking, building, grading) follows it."""
    if state.get("data_requirement"):
        return state["data_requirement"]
    io.emit("log", text="Working out what the learning objective needs from a dataset...")
    response = call_claude(
        system=DATA_REQUIREMENT_PROMPT,
        user=build_requirement_user_prompt(state["topic"], state.get("learning_objective", ""),
                                           state.get("wiki_curriculum", "")),
        max_tokens=600,
    )
    requirement = _parse_requirement(response)
    if not requirement:
        io.emit("notice", text="Could not read the data requirement; assuming labelled images "
                               "graded by accuracy.")
        requirement = dict(DEFAULT_REQUIREMENT)
    return requirement


def _generate_options(state, existing, ctx=None):
    cv = _is_cv(state)
    shown = [{k: v for k, v in o.items() if k != "verified"} for o in existing]
    response = call_claude(
        system=CV_OPTIONS_SYSTEM_PROMPT if cv else OPTIONS_SYSTEM_PROMPT,
        user=build_options_user_prompt(
            state["topic"], state.get("learning_objective", ""),
            state.get("research_output", ""), state.get("wiki_dataset_context", ""),
            json.dumps(shown, indent=2) if shown else None,
            curriculum=state.get("wiki_curriculum", ""),
            requirement=ctx["requirement"] if cv and ctx else None,
            rejected=ctx["rejected"] if cv and ctx else None,
        ),
        max_tokens=3000,
    )
    return _parse_options(response)


def _cv_options(state, existing, ctx, io):
    """Suggest datasets, check each one against the real dataset, keep those that pass, and
    feed the reasons for failures into the next round of suggestions."""
    kept = []
    for _ in range(OPTION_ROUNDS):
        batch = _generate_options(state, existing + kept, ctx)
        known = {(o.get("source"), o.get("source_id")) for o in existing + kept}
        known |= {(r["source"], r["source_id"]) for r in ctx["rejected"]}
        batch = [o for o in batch if (o.get("source"), o.get("source_id")) not in known]
        if not batch:
            continue
        io.emit("log", text=f"Checking {len(batch)} suggested datasets against the real data "
                            f"(exists, has images, labels where needed, the named classes)...")
        verify_options(batch, ctx["requirement"])
        dropped = []
        for option in batch:
            check = option["verified"]
            if check["status"] == "reject":
                reason = "; ".join(check["reasons"])
                ctx["rejected"].append({"source": option.get("source"),
                                        "source_id": option.get("source_id"), "reason": reason})
                dropped.append(f"{option.get('dataset')} (`{option.get('source_id')}`): {reason}")
            else:
                kept.append(option)
        if dropped:
            io.emit("notice", text="Dropped after checking the real dataset:\n- " + "\n- ".join(dropped))
        if len(kept) >= MIN_VERIFIED_OPTIONS:
            break
    return kept


def _normalize_requirement(current, change):
    """Apply the human's change to the data requirement; accuracy always needs labels."""
    updated = dict(current)
    if change.get("labels") in ("required", "optional", "none"):
        updated["labels"] = change["labels"]
    if change.get("grading") in ("accuracy", "structural"):
        updated["grading"] = change["grading"]
    if updated["grading"] == "accuracy":
        updated["labels"] = "required"
    if updated != current:
        updated["why"] = "set by the reviewer"
    return updated


def _recheck_options(options, ctx, io):
    """After the requirement changes, check the listed options again and drop those that no
    longer fit (their reasons join the feedback for later suggestions)."""
    verify_options(options, ctx["requirement"])
    dropped = [o for o in options if o["verified"]["status"] == "reject"]
    for option in dropped:
        reason = "; ".join(option["verified"]["reasons"])
        ctx["rejected"].append({"source": option.get("source"), "source_id": option.get("source_id"),
                                "reason": reason})
        options.remove(option)
    if dropped:
        io.emit("notice", text="Dropped because they do not fit the changed requirement:\n- "
                + "\n- ".join(f"{o.get('dataset')} (`{o.get('source_id')}`)" for o in dropped))


def _select_option(state, options, io, more, ctx=None):
    generate = not options
    while True:
        if generate:
            io.emit("notice", text="No valid options were returned. Generating another set...")
            options.extend(more(options))
            generate = False
        io.emit("options", options=options, requirement=ctx["requirement"] if ctx else None)
        if not options:
            io.emit("notice", text="Still no usable options. Ask for more options or change the topic.")
        choice = io.ask({"kind": "select_option", "n": len(options)})
        if isinstance(choice, dict) and "requirement" in choice and ctx and ctx.get("requirement"):
            ctx["requirement"] = _normalize_requirement(ctx["requirement"], choice["requirement"])
            req = ctx["requirement"]
            io.emit("log", text=f"Data requirement changed: labels {req['labels']}, graded by {req['grading']}.")
            _recheck_options(options, ctx, io)
            continue
        choice = str(choice).strip().upper()
        if choice == "M":
            new_options = more(options)
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


def _generate_design(state, selected, workspace, code, io, ctx=None):
    if _is_cv(state):
        return _generate_image_design(state, selected, workspace, code, io, ctx)
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


def _generate_image_design(state, selected, workspace, code, io, ctx=None):
    """CV datasets: download and inspect the real dataset (including where its labels live),
    let the human choose the labels and how much of it to use, build the two archives with
    fixed code in the dataset's own layout, verify them, then draft the problem.
    Returns None when the human goes back to the options."""
    ctx = ctx if ctx is not None else {"requirement": dict(DEFAULT_REQUIREMENT), "rejected": [], "layout": None}
    requirement = ctx["requirement"]
    source, source_id = selected.get("source"), selected.get("source_id")

    def reject(reason):
        ctx["rejected"].append({"source": source, "source_id": source_id, "reason": reason})
        io.emit("notice", text=f"[!] Could not use this dataset: {reason}")
        return "", ""

    io.emit("log", text=f"Downloading and inspecting {selected.get('dataset') or source_id} "
                        f"({source}). The first download of a dataset can take a few minutes...")
    try:
        info = inspect_dataset(source, source_id)
    except Exception as exc:  # noqa: BLE001 - network/source failures are reported, not fatal
        return reject(str(exc))
    needs_labels = requirement["labels"] == "required"
    if needs_labels and not any(s["labelled"] for s in info["label_sources"]):
        return reject("the objective needs labelled images, but no labels were found in the download "
                      "(no class folders and no label table)")

    defaults = default_choice(info, selected.get("classes"), use_labels=requirement["labels"] != "none")
    choice = io.ask({"kind": "image_sample", "info": info, "defaults": defaults, "requirement": requirement})
    if isinstance(choice, dict) and choice.get("back"):
        return None
    if not isinstance(choice, dict) or not choice.get("classes"):
        choice = defaults
    src = source_by_key(info, choice.get("label_source") or defaults["label_source"])
    if (needs_labels and not src["labelled"]) or (src["labelled"] and len(choice["classes"]) < 2):
        io.emit("notice", text="That choice does not fit the data requirement; using the defaults instead.")
        choice = defaults
        src = source_by_key(info, choice["label_source"])

    io.emit("log", text="Copying the chosen images (original names and folders) into the student "
                        "and hidden test archives...")
    try:
        built = build_archives(source, source_id, choice, workspace, code)
    except Exception as exc:  # noqa: BLE001 - e.g. too few readable images in a class
        io.emit("notice", text=f"[!] Could not build the image archives: {exc}")
        return "", ""
    layout = built["layout"]
    layout["grading"] = requirement["grading"] if layout["labelled"] else "structural"
    summary, problems, warnings = validate_image_dataset(workspace, code, layout)
    if problems:
        io.emit("notice", text="[!] The image dataset failed verification:\n- " + "\n- ".join(problems))
        return "", ""
    for warning in warnings:
        io.emit("notice", text=warning)
    ctx["layout"] = layout

    plan = _image_plan(info, src, choice, summary, built)
    io.emit("log", text="Writing the problem statement from the verified dataset...")
    response = call_claude(
        system=CV_PROBLEM_STATEMENT_PROMPT,
        user=build_image_statement_user_prompt(
            state["topic"], state.get("learning_objective", ""), selected,
            state.get("research_output", ""), plan, _question_reference_formats(),
            state.get("wiki_curriculum", ""), grading=layout["grading"],
        ),
        max_tokens=5000,
    )
    statement = _between(response, "---PROBLEM_STATEMENT---", "---END_PROBLEM_STATEMENT---")
    return plan, statement


def _image_plan(info, src, choice, summary, built):
    """Dataset plan built from verified facts only (no model involved)."""
    layout = built["layout"]
    n = choice["images_per_class"]
    if src["labelled"]:
        full = f"{len(src['classes'])} classes, {src['total']:,} labelled images"
        amount = "every image of each class" if choice.get("use_all") else f"{n} images per class"
        chosen = f"{len(choice['classes'])} classes, {amount}"
        held = f"{round(choice['test_fraction'] * 100)}% of each class held back at random for grading"
    else:
        full = f"{src['total']:,} images, no labels"
        chosen = "every image" if choice.get("use_all") else f"{n} images"
        held = f"{round(choice['test_fraction'] * 100)}% held back at random for grading"
    hidden = ("hidden images come from the dataset's own official test split" if built.get("official_split")
              else held)
    variant = (f"\n\n**Variant used:** `{info['variant_used']}` (the download also has "
               f"{', '.join(v for v in info['variants'] if v != info['variant_used'])})"
               if info.get("variants") and src["kind"] == "folders" else "")
    grading = ("accuracy on the hidden labelled images" if layout["grading"] == "accuracy"
               else "structural checks of what the notebook produces (no accuracy threshold)")
    verified = "\n".join(f"- {line}" for line in describe(summary).splitlines())
    layout_lines = "\n".join(f"    {line.strip()}" if line.startswith("  - ") else f"- {line}"
                             for line in describe_layout(layout).splitlines())
    return (
        f"**Source:** {info['source']} · `{info['source_id']}` ({info['name']})\n\n"
        f"**Labels:** {src['title']}\n\n"
        f"**Full dataset:** {full}{variant}\n\n"
        f"**Chosen:** {chosen}; {hidden}; duplicate and unreadable images skipped; images keep their "
        f"original names, folders, size and format\n\n"
        f"**Grading:** {grading}\n\n"
        f"**Layout (verified):**\n\n{layout_lines}\n\n"
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

    cv = _is_cv(state)
    ctx = {"requirement": _data_requirement(state, io) if cv else None, "rejected": [], "layout": None}
    if cv:
        req = ctx["requirement"]
        io.emit("log", text=f"Data requirement: labels {req['labels']}, graded by {req['grading']}. "
                            f"{req.get('needs', '')}")

    def more(existing):
        return _cv_options(state, existing, ctx, io) if cv else _generate_options(state, existing)

    options = list(state.get("options_table") or state.get("problem_dataset_options") or [])
    if not options:
        options.extend(more(options))

    while True:
        selected = _select_option(state, options, io, more, ctx)
        if selected is None:
            return {
                "options_table": options,
                "problem_dataset_options": options,
                "selected_option": None,
                "data_requirement": ctx["requirement"],
                "approved": False,
                "current_stage": "problem_options_rejected",
            }

        io.emit("log", text="Phase 2: obtaining and inspecting the selected dataset...")
        design = _generate_design(state, selected, workspace, code, io, ctx)
        if design is None:                      # Back to options from the image panel
            continue
        plan, statement = design
        if not plan or not statement:
            io.emit("notice", text="[!] Dataset preparation did not complete for this option "
                                   "(see the message above).")
            recovery = io.ask({"kind": "recovery", "assignment_type": state.get("assignment_type")})
            if recovery == "R":
                design = _generate_design(state, selected, workspace, code, io, ctx)
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
                    "data_requirement": ctx["requirement"],
                    "dataset_layout": ctx["layout"] if cv else None,
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
