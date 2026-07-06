"""
naming.py

Single source of truth for assignment file naming.

Each assignment gets its own output folder (named by topic slug) plus a short
ASSIGNMENT CODE (e.g. SLR, REG, BAG, KM). The code suffixes only the files
that travel out of the folder (uploaded to S3 / the loading sheet / zipped):
datasets, ground truth, and the deliverable zips.

Files that pytest / the platform require by exact name are NEVER suffixed:
    conftest.py, pytest.ini, requirements.txt
They live inside the isolated folder, so they never collide anyway.

Every agent builds filenames from here so the dataset writer (Agent 2), the
notebook loader, and the conftest reader all agree on the same names.
"""

import os
import re


def make_code(topic: str, provided: str = None) -> str:
    """
    Normalize a user-provided short code, or derive one from the topic.

    Provided codes win (uppercased, alphanumerics only). If none is given,
    fall back to the initials of the topic words, capped at 4 chars.
    """
    if provided:
        code = re.sub(r"[^A-Za-z0-9]+", "", provided).upper()
        if code:
            return code[:6]
    words = re.sub(r"[^a-z0-9\s]+", " ", (topic or "").lower()).split()
    code = "".join(w[0] for w in words).upper()[:4]
    return code or "ASG"


def asset_relnames(code: str) -> dict:
    """Suffixed filenames as referenced from the pytest root (used in prompts,
    the notebook load lines, and the conftest ground-truth read)."""
    return {
        "dataset_train": f"dataset_train_{code}.csv",
        "dataset_test": f"dataset_test_{code}.csv",
        "ground_truth": f"tests/ground_truth_{code}.csv",
    }


def asset_paths(workspace: str, code: str) -> dict:
    """Absolute paths for the three suffixed data files in a workspace."""
    return {
        "dataset_train": os.path.join(workspace, f"dataset_train_{code}.csv"),
        "dataset_test": os.path.join(workspace, f"dataset_test_{code}.csv"),
        "ground_truth": os.path.join(workspace, "tests", f"ground_truth_{code}.csv"),
    }


def filename_instructions(code: str) -> str:
    """Explicit, unambiguous filename block injected into agent prompts so the
    model writes/reads the suffixed names instead of generic ones."""
    names = asset_relnames(code)
    return (
        "EXACT DATA FILENAMES (use these precisely, never generic names, no S3 URLs):\n"
        f"- training data (includes target)  -> {names['dataset_train']}  (workspace root)\n"
        f"- test data (features only)        -> {names['dataset_test']}  (workspace root)\n"
        f"- held-out ground truth (targets)  -> {names['ground_truth']}\n"
        "conftest.py, pytest.ini, requirements.txt keep their exact generic names."
    )


def zip_names(code: str) -> dict:
    """Deliverable zip names (these leave the folder, so they are suffixed)."""
    return {
        "prefilled": f"PREFILLED_CODES_{code}.zip",
        "tests": f"TESTS_{code}.zip",
        "question": f"IDE_BASED_QUESTION_{code}.zip",
    }
