"""
agents/wiki_loader.py

AGENT 0 — Wiki Loader.

Not an LLM call. Pure file I/O + filtering. Runs first in the graph,
before Research, and hands every downstream agent exactly the context
slice it needs from the knowledge/ folder.

This is the "llmwiki" piece: simple file routing, no embeddings, no
vector DB. Tuned by editing the .md files in knowledge/, not by changing
this code.
"""

import os
import json
import glob
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from naming import make_code

KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge")
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def make_slug(topic: str) -> str:
    """Return a stable, filesystem-safe assignment slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower().strip()).strip("_")
    return slug or "assignment"


def _prepare_output_dir(topic: str) -> str:
    output_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "outputs", make_slug(topic)))
    os.makedirs(os.path.join(output_dir, "tests"), exist_ok=True)
    return output_dir


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_past_assignments(config_type: str, assignment_type: str) -> str:
    """
    Loads only the relevant past-assignment markdown files:
    - filtered by config_type (vscode_type / code_editor_type)
    - within that, prioritizes matching assignment_type, but includes
      all of them since the full list is still small (this is the
      filter hook to tighten later as the wiki grows)
    """
    folder = os.path.join(KNOWLEDGE_DIR, "assignments", config_type)
    files = sorted(glob.glob(os.path.join(folder, "*.md")))
    files = [f for f in files if not os.path.basename(f).upper().startswith("README")]

    if not files:
        return f"[No past {config_type} assignments found in wiki yet.]"

    same_type = []
    other_type = []

    for fpath in files:
        content = _read(fpath)
        # crude but effective: check the "Assignment category:" line
        if f"category:** {assignment_type}" in content:
            same_type.append(content)
        else:
            other_type.append(content)

    parts = []
    if same_type:
        parts.append(f"## Past {assignment_type} assignments (same type as current request)\n")
        parts.extend(same_type)
    if other_type:
        parts.append(f"\n## Other past assignments (different type, for dataset-reuse awareness only)\n")
        parts.extend(other_type)

    return "\n\n---\n\n".join(parts)


def _load_skills() -> str:
    folder = os.path.join(KNOWLEDGE_DIR, "skills")
    files = sorted(glob.glob(os.path.join(folder, "*.md")))
    return "\n\n---\n\n".join(_read(f) for f in files)


def _load_instructions() -> str:
    folder = os.path.join(KNOWLEDGE_DIR, "instructions")
    files = sorted(glob.glob(os.path.join(folder, "*.md")))
    return "\n\n---\n\n".join(_read(f) for f in files)


def _load_dataset_library(assignment_type: str) -> str:
    path = os.path.join(KNOWLEDGE_DIR, "dataset_library.json")
    with open(path, "r", encoding="utf-8") as f:
        lib = json.load(f)

    relevant = lib.get(assignment_type, [])
    return json.dumps(relevant, indent=2)


def _load_reference_formats() -> str:
    folder = os.path.join(KNOWLEDGE_DIR, "reference_formats")
    if not os.path.isdir(folder):
        return "[No reference formats found in wiki yet.]"
    files = sorted(glob.glob(os.path.join(folder, "*.md")))
    files = [f for f in files if not os.path.basename(f).upper().startswith("README")]
    if not files:
        return "[No reference format files found.]"
    return "\n\n---\n\n".join(_read(f) for f in files)


EXAMPLE_FILES = ("question.json", "conftest.py", "test_solution.py")
_EXAMPLE_LANG = {".json": "json", ".py": "python"}


def _example_dirs(config_type: str) -> list:
    """Assignment subfolders under examples/{config_type}/ (README etc. skipped
    automatically since we only look at directories)."""
    folder = os.path.join(KNOWLEDGE_DIR, "examples", config_type)
    if not os.path.isdir(folder):
        return []
    return sorted(d for d in glob.glob(os.path.join(folder, "*")) if os.path.isdir(d))


def _load_examples(config_type: str) -> str:
    """
    Loads complete worked-example assignments for this config_type. Each
    assignment is one subfolder holding the three grading-contract files
    (question.json, conftest.py, test_solution.py) as real files. Each folder
    is assembled into one labeled bundle so the file-generating agent can see
    how the pieces align.

    Loads all examples for now; add assignment_type/task_type filtering once
    the count grows past ~5.
    """
    dirs = _example_dirs(config_type)
    if not dirs:
        return "[No worked examples found in wiki yet.]"

    bundles = []
    for d in dirs:
        name = os.path.basename(d)
        parts = [f"# Example: {name}"]
        found_any = False
        for fname in EXAMPLE_FILES:
            fpath = os.path.join(d, fname)
            if os.path.isfile(fpath):
                content = _read(fpath)
                if not content.strip():
                    continue  # skip empty placeholder files
                found_any = True
                lang = _EXAMPLE_LANG.get(os.path.splitext(fname)[1], "")
                parts.append(f"## {fname}\n```{lang}\n{content}\n```")
        if found_any:
            bundles.append("\n\n".join(parts))

    if not bundles:
        return "[No worked example files found yet.]"
    return "\n\n---\n\n".join(bundles)


# ══════════════════════════════════════════════════════════════════════════
# The node function — this is what LangGraph calls
# ══════════════════════════════════════════════════════════════════════════

def wiki_loader_agent(state: dict) -> dict:
    """
    Reads state['assignment_type'] and state['config_type'],
    returns assembled context strings for downstream agents to consume.
    """
    assignment_type = state["assignment_type"]
    config_type = state.get("config_type", "vscode_type")
    output_dir = state.get("output_dir") or _prepare_output_dir(state["topic"])
    state["output_dir"] = output_dir
    assignment_code = make_code(state["topic"], state.get("assignment_code"))
    state["assignment_code"] = assignment_code

    print("\n" + "=" * 60)
    print("📚  AGENT 0: WIKI LOADER")
    print("=" * 60)
    print(f"Loading context for: {assignment_type} / {config_type}  (code: {assignment_code})")

    research_context = _load_past_assignments(config_type, assignment_type)
    skill_context = _load_skills()
    instructions_context = _load_instructions()
    dataset_context = _load_dataset_library(assignment_type)
    reference_formats_context = _load_reference_formats()
    examples_context = _load_examples(config_type)

    skill_file_count = len(glob.glob(os.path.join(KNOWLEDGE_DIR, "skills", "*.md")))
    instr_file_count = len(glob.glob(os.path.join(KNOWLEDGE_DIR, "instructions", "*.md")))
    past_file_count = len(glob.glob(os.path.join(KNOWLEDGE_DIR, "assignments", config_type, "*.md")))
    dataset_entry_count = dataset_context.count('"name"')
    ref_format_count = len(glob.glob(os.path.join(KNOWLEDGE_DIR, "reference_formats", "*.md")))
    example_count = len(_example_dirs(config_type))

    print(f"  Past assignments loaded : {past_file_count} files")
    print(f"  Skill docs loaded       : {skill_file_count} files")
    print(f"  Instructions loaded     : {instr_file_count} files ({len(instructions_context)} chars)")
    print(f"  Dataset library entries : {dataset_entry_count}")
    print(f"  Reference formats loaded: {ref_format_count} files")
    print(f"  Worked examples loaded  : {example_count} files")

    return {
        "wiki_research_context": research_context,
        "wiki_skill_context": skill_context,
        "wiki_instructions_context": instructions_context,
        "wiki_dataset_context": dataset_context,
        "wiki_reference_formats": reference_formats_context,
        "wiki_examples": examples_context,
        "output_dir": output_dir,
        "assignment_code": assignment_code,
        "current_stage": "wiki_loaded",
    }
