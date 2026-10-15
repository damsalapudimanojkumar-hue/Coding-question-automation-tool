"""
ui/checkpoints.py

LangGraph checkpointing for pipeline runs, so a run survives an app restart.

LangGraph saves the full pipeline state after every completed step (node) into
one SQLite file, keyed by the run's id. After a restart, invoking the graph with
the same run id and no input continues from the step that had not finished yet:
completed steps (and everything the human approved in them) are kept, and only the
interrupted step runs again from its start.

Questions asked INSIDE a step are not checkpointed individually, so a step that was
interrupted mid-way starts over (e.g. its options are generated again).
"""

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", ".sessions", "checkpoints.sqlite")

STEP_LABELS = {
    "wiki": "Wiki",
    "research": "Research",
    "vs_problem": "Problem & Data",
    "vs_eval": "Evaluation",
    "ce_design": "Problem",
    "ce_generate": "Deliverable",
}


def open_saver() -> SqliteSaver:
    os.makedirs(os.path.dirname(_DB), exist_ok=True)
    # The pipeline runs in a worker thread, so the connection must be shareable.
    return SqliteSaver(sqlite3.connect(_DB, check_same_thread=False))


def run_config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def next_steps(graph, run_id: str) -> tuple:
    """Steps still to run for a saved run; empty if nothing was saved or it finished."""
    try:
        return tuple(graph.get_state(run_config(run_id)).next)
    except Exception:  # noqa: BLE001 - a missing/corrupt checkpoint just means "cannot resume"
        return ()


def delete(run_id: str) -> None:
    try:
        open_saver().delete_thread(run_id)
    except Exception:  # noqa: BLE001 - cleanup must never break the app
        pass
