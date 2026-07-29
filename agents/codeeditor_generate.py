"""
Agent 3 (code-editor mode) — Generate

DETERMINISTIC. No LLM. Takes the approved question config (from the code-editor
design agent) and runs tools/testcase_generator.py to produce the platform
deliverable zip (questions.json + question_sets_questions.json).

The generator computes each test case's OUTPUT by running the config's
solution_code, then base64-encodes the boilerplate evaluator (main.py). We
surface the computed outputs back to the user so they can sanity-check them
against the design-time formula computation (the two must agree).

Input it reads from state:
    state["codeeditor_config"]  -> approved config dict (single) OR list of configs
    state["output_dir"]         -> unique per-run output folder (from Agent 0)
    state["assignment_code"]    -> short code (e.g. BCE) suffixing the deliverable
    state["topic"]

Output it writes back into state:
    state["codeeditor_deliverable"] -> path to the generated .zip
    state["final_zips"]             -> {zip_name: path} (parity with vscode flow)
"""

import sys
import os
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.testcase_generator import TestCaseGenerator, generate_bundle
from naming import zip_names, make_code
from tracing import observe


def _summarize(gen: "TestCaseGenerator") -> list:
    """Pull the computed (input, output, hidden) rows out of a built generator."""
    return [
        {
            "order": tc["order"],
            "input": tc["input"],
            "output": tc["output"],
            "is_hidden": tc["is_hidden"],
            "weightage": tc["weightage"],
        }
        for tc in gen.test_cases
    ]


def _write_readable(output_dir: str, configs: list) -> list:
    """Drop human-readable copies next to the zip so the reference solution,
    starter, and statement can be read without opening the deliverable. One set
    per question (prefixed q1_/q2_ when there is more than one)."""
    written = []
    multi = len(configs) > 1
    for i, cfg in enumerate(configs, 1):
        prefix = f"q{i}_" if multi else ""
        files = {
            f"{prefix}reference_solution.py": cfg.get("solution_code", ""),
            f"{prefix}starter_code.py": cfg.get("starter_code", ""),
            f"{prefix}question.md": cfg.get("rephrased_question_text")
                                    or cfg.get("question_text", ""),
        }
        for name, content in files.items():
            path = os.path.join(output_dir, name)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content or "")
            written.append(name)
    return written


@observe(as_type="agent")
def codeeditor_generate_agent(state: dict, io) -> dict:
    config = state.get("codeeditor_config")
    if not config:
        raise ValueError("codeeditor_generate: state['codeeditor_config'] is empty — "
                         "the design agent must run and be approved first.")

    output_dir = state.get("output_dir") or os.getcwd()
    code = state.get("assignment_code") or make_code(state.get("topic", ""))
    zip_path = os.path.join(output_dir, zip_names(code)["question"])

    io.emit("stage", text="AGENT 3 (code-editor): GENERATE")
    io.emit("log", text=f"Building deliverable -> {os.path.basename(zip_path)}")

    # A config may be a single dict (one question) or a list (a bundle).
    configs = config if isinstance(config, list) else [config]

    summaries = []
    if len(configs) == 1:
        cfg = copy.deepcopy(configs[0])
        test_definitions = cfg.pop("test_definitions", [])
        gen = TestCaseGenerator(cfg)
        if test_definitions:
            gen.add_test_cases_bulk(test_definitions)
        gen.generate(zip_path)
        summaries.append({"short_text": cfg.get("short_text", ""),
                          "function_name": cfg.get("function_name", ""),
                          "cases": _summarize(gen)})
    else:
        # Bundle mode: generate_bundle pops test_definitions internally, so build
        # per-question summaries first (on copies) before it consumes them.
        for cfg in configs:
            c = copy.deepcopy(cfg)
            tds = c.pop("test_definitions", [])
            g = TestCaseGenerator(c)
            if tds:
                g.add_test_cases_bulk(tds)
            summaries.append({"short_text": c.get("short_text", ""),
                              "function_name": c.get("function_name", ""),
                              "cases": _summarize(g)})
        generate_bundle(copy.deepcopy(configs), zip_path)

    readable = _write_readable(output_dir, configs)
    io.emit("log", text="Wrote readable copies: " + ", ".join(readable))

    io.emit("codeeditor_outputs", questions=summaries, path=zip_path, readable=readable)
    io.emit("done", text=f"Deliverable written: {zip_path}")

    return {
        "codeeditor_deliverable": zip_path,
        "codeeditor_readable_files": readable,
        "final_zips": {os.path.basename(zip_path): zip_path},
        "current_stage": "codeeditor_generated",
    }
