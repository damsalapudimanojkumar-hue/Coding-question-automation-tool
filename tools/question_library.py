"""Small, deterministic helpers for the Question Library screen.

The GitHub layer fetches saved question folders; this module turns their JSON into
display rows and combines selected code-editor questions into one platform-ready
bundle. Keeping it separate from Streamlit makes the merge rules easy to test.
"""

from __future__ import annotations

import io
import json
import uuid
import zipfile
from typing import Any


def records_from_folder(folder_path: str, files: dict[str, bytes]) -> list[dict[str, Any]]:
    """Extract one library record per question from a saved code-editor folder.

    A folder may contain a future multi-question bundle, so this intentionally
    supports more than one item in ``coding_questions.json``.
    """
    raw = files.get("coding_questions.json")
    if not raw:
        return []
    questions = json.loads(raw.decode("utf-8"))
    if not isinstance(questions, list):
        raise ValueError("coding_questions.json must contain a JSON list.")

    records = []
    for index, question in enumerate(questions, 1):
        meta = question.get("question", {}) if isinstance(question, dict) else {}
        question_id = meta.get("question_id")
        if not question_id:
            raise ValueError(f"Question #{index} has no question_id.")
        records.append({
            "key": f"{folder_path}::{question_id}",
            "folder_path": folder_path,
            "question": question,
            "question_id": question_id,
            "title": meta.get("short_text") or "Untitled question",
            "difficulty": meta.get("difficulty") or "Unknown",
            "library": _library_name(question),
        })
    return records


def build_question_set_bundle(records: list[dict[str, Any]]) -> dict[str, bytes]:
    """Create the two platform JSON files and a ZIP from selected library records.

    A fresh question_set_id is intentional: the selected questions become a new
    set without changing the source questions stored in GitHub.
    """
    if not records:
        raise ValueError("Select at least one question.")

    ids = [record["question_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("The same question was selected more than once.")

    questions = [record["question"] for record in records]
    question_set_id = str(uuid.uuid4())
    mappings = [
        {"question_set_id": question_set_id, "question_id": record["question_id"], "order": order}
        for order, record in enumerate(records, 1)
    ]
    coding_json = json.dumps(questions, indent=2).encode("utf-8")
    mappings_json = json.dumps(mappings, indent=2).encode("utf-8")

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("coding_questions.json", coding_json)
        zf.writestr("question_sets_questions.json", mappings_json)

    return {
        "coding_questions.json": coding_json,
        "question_sets_questions.json": mappings_json,
        "question_set.zip": archive.getvalue(),
    }


def _library_name(question: dict[str, Any]) -> str:
    """Read the execution library from the platform question object."""
    details = question.get("coding_question_details") or []
    if details and isinstance(details[0], dict):
        return details[0].get("language") or "Python"
    return "Python"
