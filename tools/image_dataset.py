"""Deterministic checks for a CV assignment's image dataset.

Two archives, laid out the way the dataset itself is (see tools.image_layout):
    images_<CODE>.zip              the student images (+ the dataset's own label table, if any)
    tests/test_images_<CODE>.zip   the hidden grading images (graders only)

The dataset step is fixed code, but "it says it saved real images" is still not proof.
This module extracts both archives and reads them with the same reader the grading code
uses, then reports what is really there.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from collections import Counter

from PIL import Image

from naming import images_zip_name, test_images_zip_relname
from tools.image_layout import IMAGE_EXTS, iter_dataset

SIZE_WARNING_MB = 10
UNLABELLED = "(no label)"


def archive_paths(workspace: str, code: str) -> dict:
    return {
        "train": os.path.join(workspace, images_zip_name(code)),
        "test": os.path.join(workspace, *test_images_zip_relname(code).split("/")),
    }


def default_layout(code: str) -> dict:
    """Layout of archives built before layouts were recorded: class folders under train/, test/."""
    return {"label_source": "folders", "labelled": True, "class_order": [],
            "student": {"kind": "folders", "roots": ["train"]},
            "hidden": {"kind": "folders", "roots": ["test"]},
            "student_archive": images_zip_name(code), "hidden_archive": test_images_zip_relname(code)}


def _scan(archive: str, spec: dict, problems: list[str]) -> tuple[Counter, Counter, set]:
    """Return (per-label counts, image sizes, content hashes) for one archive, read through
    the layout spec. Images the spec does not reach, and spec entries that are not readable
    images, are problems."""
    per_label, sizes, hashes = Counter(), Counter(), set()
    label = os.path.basename(archive)
    tmp = tempfile.mkdtemp(prefix="imgcheck_")
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        on_disk = {os.path.normcase(os.path.join(d, f)) for d, _, files in os.walk(tmp)
                   for f in files if f.lower().endswith(IMAGE_EXTS)}
        reached, unreadable, missing = set(), [], []
        try:
            entries = list(iter_dataset(tmp, spec))
        except Exception as exc:  # noqa: BLE001 - a layout that cannot be read is a problem
            problems.append(f"{label}: could not be read with its layout ({exc})")
            return per_label, sizes, hashes
        for path, name in entries:
            rel = os.path.relpath(path, tmp).replace(os.sep, "/") if path else "?"
            if not path or not os.path.isfile(path):
                missing.append(rel)
                continue
            reached.add(os.path.normcase(path))
            try:
                with Image.open(path) as img:
                    img.load()
                    sizes[img.size] += 1
            except Exception:  # noqa: BLE001 - any decode failure means not a real image
                unreadable.append(rel)
                continue
            per_label[UNLABELLED if name is None else name] += 1
            with open(path, "rb") as fh:
                hashes.add(hashlib.sha1(fh.read()).hexdigest())
        orphans = sorted(os.path.relpath(p, tmp) for p in on_disk - reached)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if missing:
        problems.append(f"{label}: {len(missing)} listed images are not in the archive, e.g. {missing[:3]}")
    if unreadable:
        problems.append(f"{label}: {len(unreadable)} files are not readable images, e.g. {unreadable[:3]}")
    if orphans:
        problems.append(f"{label}: {len(orphans)} images are not covered by the layout "
                        f"(no label for them), e.g. {orphans[:3]}")
    return per_label, sizes, hashes


def validate_image_dataset(workspace: str, code: str, layout: dict = None) -> tuple[dict, list[str], list[str]]:
    """Return (summary, problems, warnings). Any problem means the dataset is unusable."""
    layout = layout or default_layout(code)
    paths = archive_paths(workspace, code)
    missing = [os.path.relpath(p, workspace) for p in paths.values() if not os.path.isfile(p)]
    if missing:
        return {}, [f"Image archive not created: {', '.join(missing)}"], []

    problems: list[str] = []
    warnings: list[str] = []
    train, train_sizes, train_hashes = _scan(paths["train"], layout["student"], problems)
    test, test_sizes, test_hashes = _scan(paths["test"], layout["hidden"], problems)

    if not train:
        problems.append("The student archive has no images.")
    if not test:
        problems.append("The hidden test archive has no images.")
    if layout["labelled"] and set(train) != set(test):
        problems.append(f"Student and hidden classes differ: only in student {sorted(set(train) - set(test))}, "
                        f"only in hidden {sorted(set(test) - set(train))}")
    if layout["labelled"] and layout.get("class_order") and set(train) != set(layout["class_order"]):
        problems.append("The classes found do not match the recorded class order.")
    shared = train_hashes & test_hashes
    if shared:
        problems.append(f"{len(shared)} hidden images are exact copies of student images.")

    sizes = train_sizes + test_sizes
    archive_mb = {k: os.path.getsize(p) / (1024 * 1024) for k, p in paths.items()}
    total_mb = sum(archive_mb.values())
    if total_mb > SIZE_WARNING_MB:
        warnings.append(f"Image archives total {total_mb:.1f} MB (platform guideline is under "
                        f"{SIZE_WARNING_MB} MB).")

    order = layout.get("class_order") or sorted(train)
    summary = {
        "labelled": layout["labelled"],
        "classes": order if layout["labelled"] else [],
        "train_per_class": {c: train[c] for c in order if c in train},
        "test_per_class": {c: test[c] for c in order if c in test},
        "size_range": _size_range(sizes),
        "train_mb": round(archive_mb["train"], 2),
        "test_mb": round(archive_mb["test"], 2),
    }
    return summary, problems, warnings


def _size_range(sizes: Counter) -> str:
    if not sizes:
        return "unknown"
    smallest = min(sizes, key=lambda s: s[0] * s[1])
    largest = max(sizes, key=lambda s: s[0] * s[1])
    if smallest == largest:
        return f"{smallest[0]}x{smallest[1]} (all images)"
    return f"{smallest[0]}x{smallest[1]} to {largest[0]}x{largest[1]} (original sizes, not resized)"


def describe(summary: dict) -> str:
    """One readable block of verified facts, used in notices and recovery prompts."""
    if not summary:
        return ""
    if summary.get("labelled", True):
        head = (f"Classes ({len(summary['classes'])}): {', '.join(map(str, summary['classes']))}\n"
                f"Student images per class: {summary['train_per_class']}\n"
                f"Hidden test images per class: {summary['test_per_class']}\n")
    else:
        head = (f"No labels. Student images: {sum(summary['train_per_class'].values())}, "
                f"hidden test images: {sum(summary['test_per_class'].values())}\n")
    return (head + f"Image sizes: {summary['size_range']}\n"
            f"Archives: student {summary['train_mb']} MB, hidden test {summary['test_mb']} MB")


def _holds_images(folder: str) -> bool:
    for _, _, files in os.walk(folder):
        if any(f.lower().endswith(IMAGE_EXTS) for f in files):
            return True
    return False


def remove_extracted_images(workspace: str) -> list[str]:
    """Delete folders of unzipped images left behind by trial runs (the zips are the
    real copies). Only image-holding folders at the workspace root or directly in
    tests/ are touched. Returns the removed paths, relative to the workspace."""
    removed = []
    candidates = [os.path.join(workspace, d) for d in os.listdir(workspace) if d != "tests"]
    tests_dir = os.path.join(workspace, "tests")
    if os.path.isdir(tests_dir):
        candidates += [os.path.join(tests_dir, d) for d in os.listdir(tests_dir)]
    for path in candidates:
        if os.path.isdir(path) and _holds_images(path):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(os.path.relpath(path, workspace))
    return removed
