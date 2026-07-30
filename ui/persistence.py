"""
ui/persistence.py

Lightweight autosave for a run, so a browser refresh or a server crash doesn't
wipe an in-flight assignment. We write a compact JSON snapshot (meta + emitted
events + status) to outputs/.sessions/<run_id>.json at each human-in-the-loop
pause and on completion. The start screen can then surface an interrupted run and
let you recover the design it had already produced.

Orchestration-layer only — it never touches the agents. Every call is wrapped so
a persistence failure can never break a run.
"""

import os
import json

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "outputs", ".sessions")


def _path(run_id: str) -> str:
    return os.path.join(_DIR, f"{run_id}.json")


def save_run(run_id: str, data: dict) -> None:
    try:
        os.makedirs(_DIR, exist_ok=True)
        with open(_path(run_id), "w", encoding="utf-8") as f:
            json.dump(data, f, default=str)
    except Exception:  # noqa: BLE001 - autosave must never crash the run
        pass


def load_run(run_id: str):
    try:
        with open(_path(run_id), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def list_runs() -> list:
    """All saved snapshots, newest first."""
    if not os.path.isdir(_DIR):
        return []
    out = []
    for name in os.listdir(_DIR):
        if name.endswith(".json"):
            d = load_run(name[:-5])
            if d:
                out.append(d)
    out.sort(key=lambda d: d.get("ts", 0), reverse=True)
    return out


def delete_run(run_id: str) -> None:
    try:
        os.remove(_path(run_id))
    except OSError:
        pass
