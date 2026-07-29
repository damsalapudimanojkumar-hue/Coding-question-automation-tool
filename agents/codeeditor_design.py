"""
Agent 2 (code-editor mode) — Design  (two-phase, HITL, with options)

Options read from state:
    codeeditor_difficulty       "EASY"|"MEDIUM"|"HARD"|"" (auto)
    codeeditor_num_tests         int, target number of test cases (default 10)
    codeeditor_num_candidates    int, how many problem candidates to offer (default 1)
    codeeditor_include_examples  bool, append an Examples section (default True)

Phase 1 (PROBLEM): the model writes the statement + signature + reference solution as RAW
    fenced blocks (plain-text formulas, no LaTeX). If num_candidates > 1 the user PICKS one
    of N candidates; otherwise the user approves/revises/regenerates a single one.

Phase 2 (TESTS): the model writes test-case INPUTS as JSON. We run the reference solution to
    COMPUTE outputs, show input->output rows, and the user approves/revises/regenerates.

Finally, if enabled, an Examples section built from the first visible cases (with their real
computed outputs) is appended to the description.
"""

import sys
import os
import re
import json
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_client import call_claude
from prompts.codeeditor_design_prompt import (
    CODEEDITOR_PROBLEM_SYSTEM_PROMPT, build_codeeditor_problem_prompt, _difficulty_rule,
    CODEEDITOR_TESTS_SYSTEM_PROMPT, build_codeeditor_tests_prompt,
)
from tools.testcase_generator import TestCaseGenerator
from tracing import observe

_PROBLEM_RE = re.compile(
    r"---QUESTION_MD---\s*(?P<q>.*?)\s*"
    r"---SOLUTION_PY---\s*(?P<sol>.*?)\s*"
    r"---STARTER_PY---\s*(?P<start>.*?)\s*"
    r"---META_JSON---\s*(?P<meta>.*?)\s*"
    r"(?:---END_DESIGN---|\Z)",
    re.DOTALL,
)
_TESTS_START = re.compile(r"---TESTS_JSON---", re.IGNORECASE)
_TESTS_END = re.compile(r"---END_TESTS---", re.IGNORECASE)

# Distinct SCOPES that force each option to be a genuinely different question (not a
# reworded copy). Slot order = increasing scope; difficulty is a hint used only when the
# user leaves difficulty on "auto".
_OPTION_SCOPES = [
    {"label": "core sub-step (simplest)",
     "focus": ("Isolate the CORE sub-computation as a SMALLER, simpler standalone function - "
               "e.g. a single time-step, one gate, or the elementary operation. It must be a "
               "DIFFERENT, smaller function than the full version (different function name)."),
     "difficulty": "EASY"},
    {"label": "standard full version",
     "focus": ("The standard, full/canonical version of the concept for this topic."),
     "difficulty": "MEDIUM"},
    {"label": "extended / harder variant",
     "focus": ("A harder EXTENDED variant of the full version - add a realistic complication, an "
               "extra step, or stacking. A genuinely DIFFERENT function from the full version."),
     "difficulty": "HARD"},
    {"label": "sibling concept (same family)",
     "focus": ("A closely related but DIFFERENT operation in the same family (a sibling concept), "
               "with its own distinct function - not the main concept itself."),
     "difficulty": "MEDIUM"},
    {"label": "numerical-stability / edge focus",
     "focus": ("A variant centered on a numerical-stability or tricky edge-case requirement "
               "(e.g. overflow, underflow, clipping, empty or degenerate input), with its own "
               "distinct function."),
     "difficulty": "HARD"},
]


def _strip_fence(blob: str) -> str:
    fence = re.match(r"```(?:json|python)?\s*(.*?)\s*```", blob.strip(), re.DOTALL)
    return fence.group(1).strip() if fence else blob.strip()


def _loads_lenient(blob: str) -> dict:
    blob = _strip_fence(blob)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        start, end = blob.find("{"), blob.rfind("}")
        if start != -1 and end > start:
            return json.loads(blob[start:end + 1])
        raise


def _parse_problem(text: str) -> dict:
    m = _PROBLEM_RE.search(text or "")
    if not m:
        raise ValueError("problem envelope markers not found")
    config = dict(_loads_lenient(m.group("meta")))
    config["question_text"] = m.group("q").strip()
    config["solution_code"] = _strip_fence(m.group("sol"))
    config["starter_code"] = _strip_fence(m.group("start"))
    return config


def _parse_tests(text: str) -> list:
    """Tolerant: take everything after the LAST start marker, drop an optional end
    marker, then load leniently. Survives reasoning-before-marker and a missing end."""
    if not text:
        raise ValueError("empty tests response")
    starts = list(_TESTS_START.finditer(text))
    if starts:
        blob = text[starts[-1].end():]
        end = _TESTS_END.search(blob)
        if end:
            blob = blob[:end.start()]
    else:
        blob = text
    obj = _loads_lenient(blob)
    return obj if isinstance(obj, list) else obj.get("test_definitions", [])


# ── validation ─────────────────────────────────────────────────────────────

_PROBLEM_REQUIRED = ["question_text", "short_text", "function_name",
                     "param_names", "starter_code", "solution_code"]
# Formula styles that break on the platform (LaTeX not rendered).
_LATEX_MARKERS = ["$$", "$", "\\frac", "\\text", "\\sum", "\\sqrt", "\\begin", "\\mathbf", "\\left"]


def _validate_problem(config: dict) -> list:
    problems = []
    for key in _PROBLEM_REQUIRED:
        if not config.get(key):
            problems.append(f"missing required field: {key}")
    fn = config.get("function_name", "")
    if fn:
        for field in ("starter_code", "solution_code"):
            code = config.get(field, "")
            if code and f"def {fn}" not in code:
                problems.append(f"function_name '{fn}' not defined in {field}")
    if not isinstance(config.get("param_names"), list) or not config.get("param_names"):
        problems.append("param_names must be a non-empty list")
    qt = config.get("question_text", "")
    if any(m in qt for m in _LATEX_MARKERS):
        problems.append("question_text contains LaTeX (won't render on the platform — use plain-text formulas)")
    if qt.lstrip().startswith("#"):
        problems.append("question_text starts with a heading (use no H1 title; start with the intro sentence)")
    return problems


def _validate_tests(test_definitions: list, param_names: list, num_tests: int = 10) -> list:
    problems = []
    if not test_definitions:
        return ["no test cases"]
    lo, hi = min(8, num_tests), max(12, num_tests)
    if not (lo <= len(test_definitions) <= hi):
        problems.append(f"test count is {len(test_definitions)} (asked for ~{num_tests})")
    total_w, visible = 0, 0
    for i, td in enumerate(test_definitions, 1):
        inputs = td.get("inputs")
        if not isinstance(inputs, dict):
            problems.append(f"test {i}: inputs must be a dict of param->value")
            continue
        for p in param_names:
            if p not in inputs:
                problems.append(f"test {i}: missing input for param '{p}'")
        if "output" in td or "expected" in td:
            problems.append(f"test {i}: must NOT carry outputs (they are computed)")
        total_w += td.get("weightage", 10)
        if not td.get("is_hidden", False):
            visible += 1
    if total_w != 100:
        problems.append(f"weightages sum to {total_w} (should be 100)")
    if not (2 <= visible <= 4):
        problems.append(f"{visible} visible cases (aim for 2-4 visible, rest hidden)")
    return problems


def _compute_cases(problem_config: dict, test_definitions: list) -> tuple:
    try:
        cfg = copy.deepcopy(problem_config)
        cfg.pop("test_definitions", None)
        gen = TestCaseGenerator(cfg)
        gen.add_test_cases_bulk(copy.deepcopy(test_definitions))
        cases = [
            {"order": tc["order"], "input": tc["input"], "output": tc["output"],
             "is_hidden": tc["is_hidden"], "weightage": tc["weightage"]}
            for tc in gen.test_cases
        ]
        return cases, None
    except Exception as e:  # noqa: BLE001
        return [], f"{type(e).__name__}: {e}"


def _example_descriptions(raw: list, k: int = 3) -> str:
    """Slice examples down to just their DESCRIPTIONS (+ signature) for the problem
    phase - no solution_code, no test JSON - so the model reads only clean prose to
    imitate (removes the ~70% code/test noise the full config would carry)."""
    blocks = []
    for c in (raw or [])[:k]:
        title = c.get("rephrased_short_text") or c.get("short_text", "")
        params = ", ".join(c.get("param_names", []))
        fn = c.get("function_name", "")
        qt = c.get("rephrased_question_text") or c.get("question_text", "")
        blocks.append(f"# {title}\nFunction: {fn}({params})\n\n{qt}")
    return "\n\n---\n\n".join(blocks) if blocks else "[no example descriptions yet]"


def _example_tests(raw: list, k: int = 2) -> str:
    """Slice examples down to just their TEST-CASE PATTERNS for the tests phase."""
    blocks = []
    for c in (raw or [])[:k]:
        fn = c.get("function_name", "")
        params = ", ".join(c.get("param_names", []))
        tds = c.get("test_definitions", []) or []
        compact = [{"inputs": t.get("inputs"),
                    "is_hidden": t.get("is_hidden", False),
                    "weightage": t.get("weightage", 10)} for t in tds]
        blocks.append(f"# {c.get('short_text','')} - {fn}({params})\n"
                      + json.dumps({"test_definitions": compact}, indent=1))
    return "\n\n".join(blocks) if blocks else ""


# ── LLM passes ───────────────────────────────────────────────────────────────

def _design_problem(state: dict, feedback: str = "", idea=None) -> tuple:
    # explicit user difficulty wins; else take the candidate idea's difficulty; else auto
    difficulty = state.get("codeeditor_difficulty", "") or (
        idea.get("difficulty", "") if isinstance(idea, dict) else "")
    system = CODEEDITOR_PROBLEM_SYSTEM_PROMPT.replace("[[DIFFICULTY_RULE]]", _difficulty_rule(difficulty))
    prompt = build_codeeditor_problem_prompt(
        topic=state["topic"],
        learning_objective=state.get("learning_objective", ""),
        research_output=state.get("research_output", ""),
        reference_docs=state.get("wiki_reference_problem") or state.get("wiki_reference_formats", ""),
        example_config=_example_descriptions(state.get("wiki_examples_raw") or []),
        difficulty=difficulty,
        idea=idea,
    )
    if feedback:
        prompt += ("\n\nREVISION REQUESTED — address this precisely and re-emit ALL four "
                   "blocks:\n" + feedback)
    raw = call_claude(system=system, user=prompt, max_tokens=6000)
    try:
        config = _parse_problem(raw)
    except Exception as e:  # noqa: BLE001
        return None, raw, [f"could not parse problem: {e}"]
    return config, raw, _validate_problem(config)


def _option_specs(state: dict, n: int) -> list:
    """Build n distinct option specs from the scope rotation. Each spec is an `idea`
    dict passed to _design_problem. If the user pinned a difficulty we keep it (options
    still differ by SCOPE); otherwise we spread EASY/MEDIUM/HARD across the scopes."""
    pinned = state.get("codeeditor_difficulty", "")
    rotated = (_OPTION_SCOPES * ((n // len(_OPTION_SCOPES)) + 1))[:n]
    return [
        {"title": "", "focus": s["focus"], "difficulty": pinned or s["difficulty"],
         "_label": s["label"]}
        for s in rotated
    ]


def _design_tests(state: dict, problem_config: dict, feedback: str = "") -> tuple:
    num_tests = int(state.get("codeeditor_num_tests", 10) or 10)
    prompt = build_codeeditor_tests_prompt(
        problem_config=problem_config,
        reference_docs=state.get("wiki_reference_tests") or state.get("wiki_reference_formats", ""),
        num_tests=num_tests,
        example_tests=_example_tests(state.get("wiki_examples_raw") or []),
    )
    if feedback:
        prompt += ("\n\nREVISION REQUESTED — address this precisely and re-emit the FULL "
                   "test_definitions JSON:\n" + feedback)
    raw = call_claude(system=CODEEDITOR_TESTS_SYSTEM_PROMPT, user=prompt, max_tokens=6000)
    try:
        tds = _parse_tests(raw)
    except Exception as e:  # noqa: BLE001
        return None, raw, [f"could not parse test cases: {e}"]
    return tds, raw, _validate_tests(tds, problem_config.get("param_names", []), num_tests)


def _review_choice(answer: str) -> str:
    c = (answer or "").strip()
    if c.upper() == "A":
        return "approve"
    if c.upper() == "R":
        return "regenerate"
    return answer


# ── the node ─────────────────────────────────────────────────────────────────

@observe(as_type="agent")
def codeeditor_design_agent(state: dict, io) -> dict:
    n_cand = int(state.get("codeeditor_num_candidates", 1) or 1)

    # ---- Phase 1: PROBLEM ----
    if n_cand > 1:
        problem = _pick_candidate(state, io, n_cand)
    else:
        problem = _single_problem(state, io)

    # ---- Phase 2: TEST CASES ----
    io.emit("stage", text="AGENT 2 (code-editor): DESIGN TEST CASES")
    io.emit("log", text="Designing test cases and computing their outputs...")
    feedback, tds, cases = "", None, []
    while True:
        tds, raw, problems = _design_tests(state, problem, feedback)
        if tds is None:
            io.emit("notice", text="[design] Could not parse test cases.\n" + raw[:1200])
            ans = io.ask({"kind": "codeeditor_review", "phase": "tests",
                          "problems": problems, "unparseable": True})
            feedback = "" if _review_choice(ans) in ("approve", "regenerate") else ans
            continue
        cases, err = _compute_cases(problem, tds)
        preview_problems = list(problems)
        if err:
            preview_problems.insert(0, f"reference solution FAILED on an input: {err}")
        full = dict(problem)
        full["test_definitions"] = tds
        io.emit("codeeditor_preview", phase="tests", config=full,
                cases=cases, problems=preview_problems)
        decision = _review_choice(io.ask({"kind": "codeeditor_review", "phase": "tests",
                                          "problems": preview_problems}))
        if decision == "approve":
            break
        feedback = "" if decision == "regenerate" else decision

    # The visible test cases ARE the worked samples (shown in the platform's test
    # panel), so we no longer append a separate Examples section to the description.
    config = dict(problem)
    config["test_definitions"] = tds
    return {"codeeditor_config": config, "current_stage": "codeeditor_designed"}


def _single_problem(state: dict, io) -> dict:
    io.emit("stage", text="AGENT 2 (code-editor): DESIGN PROBLEM")
    io.emit("log", text=f"Designing the problem for: {state['topic']}")
    feedback = ""
    while True:
        problem, raw, problems = _design_problem(state, feedback)
        if problem is None:
            io.emit("notice", text="[design] Could not parse the problem.\n" + raw[:1200])
            ans = io.ask({"kind": "codeeditor_review", "phase": "problem",
                          "problems": problems, "unparseable": True})
            feedback = "" if _review_choice(ans) in ("approve", "regenerate") else ans
            continue
        io.emit("codeeditor_preview", phase="problem", config=problem, problems=problems)
        decision = _review_choice(io.ask({"kind": "codeeditor_review", "phase": "problem",
                                          "problems": problems}))
        if decision == "approve":
            return problem
        feedback = "" if decision == "regenerate" else decision


def _pick_candidate(state: dict, io, n_cand: int) -> dict:
    io.emit("stage", text=f"AGENT 2 (code-editor): {n_cand} PROBLEM OPTIONS")
    feedback = ""
    while True:
        specs = _option_specs(state, n_cand)
        io.emit("log", text="Drafting distinct options: "
                            + ", ".join(s["_label"] for s in specs))
        cands = []
        for idea in specs:
            cfg, _, probs = _design_problem(state, feedback, idea=idea)
            if cfg:
                cands.append((cfg, probs, idea))
        if not cands:
            io.emit("notice", text="[design] No option parsed; regenerating.")
            continue
        io.emit("codeeditor_candidates", candidates=[
            {"index": i + 1,
             "short_text": c.get("rephrased_short_text") or c.get("short_text", ""),
             "function_name": c.get("function_name", ""),
             "difficulty": c.get("difficulty", ""),
             "focus": idea.get("_label", ""),
             "question_text": c.get("question_text", ""),
             "problems": p}
            for i, (c, p, idea) in enumerate(cands)
        ])
        ans = io.ask({"kind": "codeeditor_pick", "count": len(cands)})
        a = ans.strip().upper()
        if a.isdigit() and 1 <= int(a) <= len(cands):
            return cands[int(a) - 1][0]
        feedback = "" if a == "R" else ans
