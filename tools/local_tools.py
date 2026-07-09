"""
tools/local_tools.py

These are the REAL capabilities we give to agents via call_claude_with_tools().
This is what turns "Claude writes text describing a dataset" into
"Claude actually downloads/inspects a dataset and tells you what it found."

CHANGED: tool definitions converted from Anthropic's schema
({"name", "description", "input_schema"}) to OpenAI/OpenRouter's
function-calling schema ({"type": "function", "function": {"name",
"description", "parameters"}}), since we now route through OpenRouter's
OpenAI-compatible endpoint. The EXECUTOR functions below are unchanged —
only the tool definition shape changed, not what actually runs.
"""

import subprocess
import sys
import os
import json
import tempfile
import textwrap

# Cross-platform scratch dir for run_python. NOT the assignment output folder:
# agents save datasets/ground-truth using ABSOLUTE workspace paths (see the
# filename instructions in prompts); this is only for the throwaway script and
# any relative scratch the model writes. Using the OS temp dir avoids the old
# hardcoded "/home/claude/workspace" that failed with PermissionError on Windows
# (C:\home) and on locked-down deploy hosts.
_SCRATCH_DIR = os.path.join(tempfile.gettempdir(), "assignment_pipeline_ws")


# ══════════════════════════════════════════════════════════════════════════
# TOOL: run_python
# Lets the agent execute arbitrary Python code in a sandboxed subprocess.
# Used by: Dataset agent (inspect candidate datasets), Solution agent
# (verify code runs), Verification Loop (compute metrics)
# ══════════════════════════════════════════════════════════════════════════

RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute Python code in a sandboxed environment with internet access. "
            "Use this to: inspect candidate datasets (load, check shape/nulls/size), "
            "test that solution code actually runs, compute real metrics (RMSE, "
            "accuracy) instead of guessing them, verify a lazy/wrong solution "
            "actually fails. Available libraries: pandas, numpy, sklearn, torch, "
            "matplotlib (no plt.show), requests. Returns stdout + stderr. "
            "Save datasets/outputs using the ABSOLUTE workspace path given in your "
            "instructions (do not rely on the current working directory)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max execution time, default 60",
                    "default": 60
                }
            },
            "required": ["code"]
        }
    }
}


def execute_run_python(tool_input: dict) -> str:
    code = tool_input["code"]
    timeout = tool_input.get("timeout_seconds", 60)

    try:
        os.makedirs(_SCRATCH_DIR, exist_ok=True)
        script_path = os.path.join(_SCRATCH_DIR, "_agent_script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            [sys.executable, script_path],
            cwd=_SCRATCH_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n\n[Exit code: {result.returncode}]"
        return output
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT after {timeout}s] Code took too long to run."
    except Exception as e:
        return f"[EXECUTION ERROR] {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════
# TOOL: run_pytest
# Runs pytest against a given test directory + solution file.
# Used by: Verification Loop agent
# ══════════════════════════════════════════════════════════════════════════

RUN_PYTEST_TOOL = {
    "type": "function",
    "function": {
        "name": "run_pytest",
        "description": (
            "Run pytest in a given working directory and return the full output "
            "including pass/fail counts per test. Use this to verify that the "
            "reference solution.ipynb passes ALL test cases, and that a lazy/wrong "
            "solution FAILS the appropriate ones. The directory must already "
            "contain pytest.ini, tests/, and the notebook/dataset files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "working_dir": {
                    "type": "string",
                    "description": "Absolute path to the directory to run pytest in"
                }
            },
            "required": ["working_dir"]  
        }
    }
}


def execute_run_pytest(tool_input: dict) -> str:
    working_dir = tool_input["working_dir"]

    if not os.path.isdir(working_dir):
        return f"[ERROR] Directory does not exist: {working_dir}"

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-vv", "--tb=short"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "[TIMEOUT] pytest took longer than 120s"
    except Exception as e:
        return f"[EXECUTION ERROR] {str(e)}"


# ══════════════════════════════════════════════════════════════════════════
# TOOL: write_file
# Lets the agent write a file to disk (dataset CSVs, notebook JSON, etc.)
# ══════════════════════════════════════════════════════════════════════════

WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Write content to a file on disk. Use this to save dataset CSVs, "
            "the solution.ipynb (as JSON), conftest.py, test_solution.py, or "
            "any other file the pipeline needs to produce. Creates parent "
            "directories if they don't exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute file path to write to"
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write"
                }
            },
            "required": ["path", "content"]
        }
    }
}


def execute_write_file(tool_input: dict) -> str:
    path = tool_input["path"]
    content = tool_input["content"]

    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        size_kb = os.path.getsize(path) / 1024
        return f"[OK] Wrote {len(content)} chars ({size_kb:.1f} KB) to {path}"
    except Exception as e:
        return f"[WRITE ERROR] {type(e).__name__}: {e} (path={path})"


# ══════════════════════════════════════════════════════════════════════════
# TOOL: read_file
# Lets the agent read back a file it or a previous agent wrote.
# ══════════════════════════════════════════════════════════════════════════

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the full content of a file from disk.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path to read"}
            },
            "required": ["path"]
        }
    }
}


def execute_read_file(tool_input: dict) -> str:
    path = tool_input["path"]
    if not os.path.isfile(path):
        return f"[ERROR] File does not exist: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ══════════════════════════════════════════════════════════════════════════
# TOOL: list_dir
# ══════════════════════════════════════════════════════════════════════════

LIST_DIR_TOOL = {
    "type": "function",
    "function": {
        "name": "list_dir",
        "description": "List files in a directory, recursively (2 levels deep).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute directory path"}
            },
            "required": ["path"]
        }
    }
}


def execute_list_dir(tool_input: dict) -> str:
    path = tool_input["path"]
    if not os.path.isdir(path):
        return f"[ERROR] Directory does not exist: {path}"

    lines = []
    for root, dirs, files in os.walk(path):
        depth = root[len(path):].count(os.sep)
        if depth >= 2:
            dirs[:] = []
            continue
        for f in files:
            lines.append(os.path.join(root, f))
    return "\n".join(lines) if lines else "[empty directory]"


# ══════════════════════════════════════════════════════════════════════════
# Tool registry — combine definitions + executor dispatch
# ══════════════════════════════════════════════════════════════════════════

ALL_TOOLS = [RUN_PYTHON_TOOL, RUN_PYTEST_TOOL, WRITE_FILE_TOOL, READ_FILE_TOOL, LIST_DIR_TOOL]

_EXECUTORS = {
    "run_python": execute_run_python,
    "run_pytest": execute_run_pytest,
    "write_file": execute_write_file,
    "read_file": execute_read_file,
    "list_dir": execute_list_dir,
}


def tool_executor(tool_name: str, tool_input: dict) -> str:
    """Single dispatch point — passed to call_claude_with_tools()."""
    if tool_name not in _EXECUTORS:
        return f"[ERROR] Unknown tool: {tool_name}"
    return _EXECUTORS[tool_name](tool_input)