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
import uuid
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from naming import make_code
from tracing import observe

KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge")
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def make_slug(topic: str) -> str:
    """Return a stable, filesystem-safe assignment slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower().strip()).strip("_")
    return slug or "assignment"


def _prepare_output_dir(topic: str, code: str = None) -> str:
    """Create a UNIQUE folder per run so re-generating the same topic never
    overwrites a previous assignment. Name = <slug>_<code>_<timestamp>-<rand>,
    e.g. outputs/bagging_BAG_20260722-143512-a1b2/."""
    slug = make_slug(topic)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    name = "_".join(part for part in (slug, code, f"{stamp}-{suffix}") if part)
    output_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "outputs", name))
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


# ── Curriculum (taught content) — slice by topic so the 100k-token file never
#    loads whole. Matching is plain word overlap against the module headings. ──

_CURRICULUM_STOP = {"module", "the", "a", "an", "of", "to", "and", "for", "with", "in", "on", "intro"}


def _headings(lines):
    """Return (line_index, level, text) for every markdown heading."""
    out = []
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if m:
            out.append((i, len(m.group(1)), m.group(2)))
    return out


def _sig_words(text):
    words = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return {w for w in words if w and not w.isdigit() and w not in _CURRICULUM_STOP}


def _extract_module(text, topic, cap=12000):
    """Slice the section whose heading best word-matches the topic. Tries
    module-level (####) first, then unit-level (###). Returns '' if no match."""
    if not text or not topic:
        return ""
    lines = text.splitlines()
    heads = _headings(lines)
    topic_words = _sig_words(topic)
    if not topic_words:
        return ""
    for level in (4, 3):
        best_idx, best_line, best_score = None, None, 0
        for idx, (i, lvl, htext) in enumerate(heads):
            if lvl != level:
                continue
            score = len(topic_words & _sig_words(htext))
            if score > best_score:
                best_idx, best_line, best_score = idx, i, score
        if best_idx is not None:
            end = len(lines)
            for (j, lvl2, _t) in heads[best_idx + 1:]:
                if lvl2 <= level:
                    end = j
                    break
            return "\n".join(lines[best_line:end]).strip()[:cap]
    return ""


def _extract_named_section(text, needle, cap=8000):
    """Return the section under the first heading containing `needle`."""
    if not text:
        return ""
    lines = text.splitlines()
    heads = _headings(lines)
    for idx, (i, lvl, htext) in enumerate(heads):
        if needle.lower() in htext.lower():
            end = len(lines)
            for (j, lvl2, _t) in heads[idx + 1:]:
                if lvl2 <= lvl:
                    end = j
                    break
            return "\n".join(lines[i:end]).strip()[:cap]
    return ""


def _curriculum_dir():
    """Locate the curriculum folder case-insensitively (folder is 'Curriculum';
    Linux/Streamlit-Cloud is case-sensitive, so we can't hardcode the case)."""
    if not os.path.isdir(KNOWLEDGE_DIR):
        return None
    for name in os.listdir(KNOWLEDGE_DIR):
        if name.lower() == "curriculum" and os.path.isdir(os.path.join(KNOWLEDGE_DIR, name)):
            return os.path.join(KNOWLEDGE_DIR, name)
    return None


def _curriculum_text():
    folder = _curriculum_dir()
    if not folder:
        return ""
    files = [f for f in sorted(glob.glob(os.path.join(folder, "*.md")))
             if not os.path.basename(f).upper().startswith("README")]
    return "\n\n".join(_read(f) for f in files) if files else ""


def _curriculum_assignments_digest(text: str) -> str:
    """Compile the program-wide list of assignments already built, from the
    curriculum's 'Assessed by' lines (each names the topic + dataset). This is
    the cross-topic 'already used, do not reuse' list — distinct from the
    topic-only module slice."""
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("**Assessed by"):
            lines.append("- " + s.replace("**", "").strip())
    return "\n".join(lines)


def _load_curriculum(topic: str) -> str:
    text = _curriculum_text()
    if not text:
        return "[No curriculum file added yet.]"

    parts = []
    used = _curriculum_assignments_digest(text)
    if used:
        parts.append(
            "PROGRAM-WIDE CODING ASSIGNMENTS ALREADY BUILT (across ALL topics; do NOT "
            "reuse these datasets or framings):\n" + used
        )
    section = _extract_module(text, topic)
    if section:
        parts.append("TAUGHT CONTENT FOR THIS TOPIC:\n" + section)

    if not parts:
        return "[No matching curriculum module for this topic.]"
    return "\n\n---\n\n".join(parts)


def _load_eval_styles() -> str:
    return _extract_named_section(_curriculum_text(), "evaluation styles")


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


# ── Code-editor (config_type == code_editor_type) loaders ──────────────────
# Code-editor examples are single .json CONFIG files (not subfolders), and the
# reference docs live under reference_formats/code_editor_type/.

def _load_codeeditor_examples() -> str:
    folder = os.path.join(KNOWLEDGE_DIR, "examples", "code_editor_type")
    if not os.path.isdir(folder):
        return "[No code-editor examples yet.]"
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    if not files:
        return "[No code-editor example configs yet.]"
    parts = [f"# Example config: {os.path.basename(f)}\n```json\n{_read(f)}\n```" for f in files]
    return "\n\n---\n\n".join(parts)


def _load_codeeditor_reference() -> str:
    folder = os.path.join(KNOWLEDGE_DIR, "reference_formats", "code_editor_type")
    if not os.path.isdir(folder):
        return "[No code-editor reference docs yet.]"
    files = [f for f in sorted(glob.glob(os.path.join(folder, "*.md")))
             if not os.path.basename(f).upper().startswith("README")]
    return "\n\n---\n\n".join(_read(f) for f in files) if files else "[No code-editor reference docs.]"


def _load_codeeditor_examples_raw() -> list:
    """Parsed example configs, so the design agent can slice each one by phase
    (feed only the DESCRIPTION to the problem step, only the TEST PATTERNS to the
    tests step) instead of dumping the whole config into every prompt."""
    folder = os.path.join(KNOWLEDGE_DIR, "examples", "code_editor_type")
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            out.append(json.loads(_read(f)))
        except Exception:  # noqa: BLE001 - skip a malformed example, don't crash the load
            pass
    return out


def _load_codeeditor_reference_doc(filename: str) -> str:
    path = os.path.join(KNOWLEDGE_DIR, "reference_formats", "code_editor_type", filename)
    return _read(path) if os.path.isfile(path) else ""


# ══════════════════════════════════════════════════════════════════════════
# The node function — this is what LangGraph calls
# ══════════════════════════════════════════════════════════════════════════

@observe(name="wiki_loader", as_type="tool")
def wiki_loader_agent(state: dict) -> dict:
    """
    Reads state['assignment_type'] and state['config_type'],
    returns assembled context strings for downstream agents to consume.
    """
    assignment_type = state["assignment_type"]
    config_type = state.get("config_type", "vscode_type")
    assignment_code = make_code(state["topic"], state.get("assignment_code"))
    state["assignment_code"] = assignment_code
    output_dir = state.get("output_dir") or _prepare_output_dir(state["topic"], assignment_code)
    state["output_dir"] = output_dir

    print("\n" + "=" * 60)
    print("📚  AGENT 0: WIKI LOADER")
    print("=" * 60)
    print(f"Loading context for: {assignment_type} / {config_type}  (code: {assignment_code})")
    print(f"Output folder: outputs/{os.path.basename(output_dir)}")

    # ── Code-editor mode: load only its lean context (no dataset/curriculum/pytest skills) ──
    if config_type == "code_editor_type":
        examples_context = _load_codeeditor_examples()          # full string (compat / display)
        examples_raw = _load_codeeditor_examples_raw()          # parsed, for per-phase slicing
        reference_formats_context = _load_codeeditor_reference()
        ref_problem = _load_codeeditor_reference_doc("question_structure.md")  # description phase
        ref_tests = _load_codeeditor_reference_doc("test_case_design.md")      # tests phase
        research_context = _load_past_assignments(config_type, assignment_type)  # empty until built
        print(f"  Code-editor examples    : {len(examples_raw)} config(s)")
        print(f"  Code-editor reference   : problem={'ok' if ref_problem else 'MISSING'}, "
              f"tests={'ok' if ref_tests else 'MISSING'}")
        return {
            "wiki_research_context": research_context,
            "wiki_skill_context": "",
            "wiki_instructions_context": "",
            "wiki_dataset_context": "",
            "wiki_reference_formats": reference_formats_context,
            "wiki_reference_problem": ref_problem,
            "wiki_reference_tests": ref_tests,
            "wiki_examples": examples_context,
            "wiki_examples_raw": examples_raw,
            "wiki_curriculum": "",
            "wiki_eval_styles": "",
            "output_dir": output_dir,
            "assignment_code": assignment_code,
            "current_stage": "wiki_loaded",
        }

    research_context = _load_past_assignments(config_type, assignment_type)
    skill_context = _load_skills()
    instructions_context = _load_instructions()
    dataset_context = _load_dataset_library(assignment_type)
    reference_formats_context = _load_reference_formats()
    examples_context = _load_examples(config_type)
    curriculum_context = _load_curriculum(state["topic"])
    eval_styles_context = _load_eval_styles()

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
    if curriculum_context and not curriculum_context.startswith("["):
        matched = curriculum_context.splitlines()[0].lstrip("# ").strip()
        print(f"  Curriculum slice        : {matched[:55]}")
    else:
        print(f"  Curriculum slice        : (no match / no file)")

    return {
        "wiki_research_context": research_context,
        "wiki_skill_context": skill_context,
        "wiki_instructions_context": instructions_context,
        "wiki_dataset_context": dataset_context,
        "wiki_reference_formats": reference_formats_context,
        "wiki_examples": examples_context,
        "wiki_curriculum": curriculum_context,
        "wiki_eval_styles": eval_styles_context,
        "output_dir": output_dir,
        "assignment_code": assignment_code,
        "current_stage": "wiki_loaded",
    }
