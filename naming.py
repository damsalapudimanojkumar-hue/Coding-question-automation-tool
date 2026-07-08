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


# Assignment types whose "dataset" is real image files on disk (folder-per-class),
# not a tabular CSV. Kept as a set so callers can pass "cv", "image", or "vision".
_IMAGE_TYPES = {"cv", "image", "images", "vision", "computer_vision"}


def is_image_type(assignment_type: str) -> bool:
    """True when the dataset should be materialized as image files/folders
    rather than CSV tables."""
    return (assignment_type or "").strip().lower() in _IMAGE_TYPES


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


def asset_relnames(code: str, assignment_type: str = "tabular") -> dict:
    """Suffixed dataset asset names as referenced from the pytest root (used in
    prompts, the notebook load lines, and the conftest ground-truth read).

    Tabular/NLP: three CSV files (train + test + ground truth).
    Image (CV):  two DIRECTORIES (train folder-per-class + flat test folder)
                 plus a single ground-truth CSV manifest (filename,label).
    """
    if is_image_type(assignment_type):
        return {
            "dataset_train": f"dataset_train_{code}",           # directory (ImageFolder layout)
            "dataset_test": f"dataset_test_{code}",             # directory (flat, unlabeled)
            "ground_truth": f"tests/ground_truth_{code}.csv",   # filename,label manifest
        }
    return {
        "dataset_train": f"dataset_train_{code}.csv",
        "dataset_test": f"dataset_test_{code}.csv",
        "ground_truth": f"tests/ground_truth_{code}.csv",
    }


def asset_paths(workspace: str, code: str, assignment_type: str = "tabular") -> dict:
    """Absolute paths for the suffixed dataset assets in a workspace.

    For image assignments dataset_train/dataset_test are directories, not files.
    """
    names = asset_relnames(code, assignment_type)
    return {key: os.path.join(workspace, *rel.split("/")) for key, rel in names.items()}


def filename_instructions(code: str, assignment_type: str = "tabular") -> str:
    """Explicit, unambiguous asset-name block injected into agent prompts so the
    model writes/reads the suffixed names instead of generic ones."""
    names = asset_relnames(code, assignment_type)
    if is_image_type(assignment_type):
        return (
            "EXACT DATASET LAYOUT (image / computer-vision assignment). Save REAL "
            "image files on disk. NEVER flatten images into pixel columns or write a "
            "CSV of pixel values, and no S3 URLs:\n"
            f"- training images -> {names['dataset_train']}/<class_name>/<image files>  "
            "(one subfolder PER CLASS, folder name = the exact class label; this is the "
            "torchvision ImageFolder layout)\n"
            f"- test images (NO labels) -> {names['dataset_test']}/<image files>  "
            "(a FLAT folder of images the student predicts; no class subfolders)\n"
            f"- held-out ground truth -> {names['ground_truth']}  "
            "(the ONLY CSV: exactly two columns, filename,label, one row per test image)\n"
            "Students must be able to load the training data with "
            "torchvision.datasets.ImageFolder. Do NOT create "
            f"dataset_train_{code}.csv or any pixel-matrix CSV.\n"
            "conftest.py, pytest.ini, requirements.txt keep their exact generic names."
        )
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
