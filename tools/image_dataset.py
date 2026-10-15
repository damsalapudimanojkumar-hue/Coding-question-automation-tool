"""Deterministic checks for a CV assignment's image dataset.

Layout (labels come from folder names, so there are no CSV files; images keep their
original format and size):
    images_<CODE>.zip              train/<class>/<image>   (students get this)
    tests/test_images_<CODE>.zip   test/<class>/<image>    (hidden, graders only)

The dataset agent is an LLM, so "it says it saved real images" is not proof.
This module opens both archives itself and reports what is really there.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import zipfile
from collections import Counter

from PIL import Image

from naming import images_zip_name, test_images_zip_relname

SIZE_WARNING_MB = 10
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ppm", ".pgm")


def archive_paths(workspace: str, code: str) -> dict:
    return {
        "train": os.path.join(workspace, images_zip_name(code)),
        "test": os.path.join(workspace, *test_images_zip_relname(code).split("/")),
    }


def _scan(archive: str, split: str, problems: list[str]) -> tuple[Counter, Counter, set]:
    """Return (per-class counts, image sizes, content hashes) for one archive."""
    per_class, sizes, hashes = Counter(), Counter(), set()
    unreadable, misplaced = [], []
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            parts = name.split("/")
            if len(parts) != 3 or parts[0] != split:
                misplaced.append(name)
                continue
            data = zf.read(name)
            try:
                with Image.open(io.BytesIO(data)) as img:
                    img.load()
                    sizes[img.size] += 1
            except Exception:  # noqa: BLE001 - any decode failure means not a real image
                unreadable.append(name)
                continue
            per_class[parts[1]] += 1
            hashes.add(hashlib.sha1(data).hexdigest())
    label = os.path.basename(archive)
    if misplaced:
        problems.append(f"{label}: {len(misplaced)} files are not in {split}/<class>/<file>, "
                        f"e.g. {misplaced[:3]}")
    if unreadable:
        problems.append(f"{label}: {len(unreadable)} files are not readable images, "
                        f"e.g. {unreadable[:3]}")
    return per_class, sizes, hashes


def validate_image_dataset(workspace: str, code: str) -> tuple[dict, list[str], list[str]]:
    """Return (summary, problems, warnings). Any problem means the dataset is unusable."""
    paths = archive_paths(workspace, code)
    missing = [os.path.relpath(p, workspace) for p in paths.values() if not os.path.isfile(p)]
    if missing:
        return {}, [f"Image archive not created: {', '.join(missing)}"], []

    problems: list[str] = []
    warnings: list[str] = []
    train, train_sizes, train_hashes = _scan(paths["train"], "train", problems)
    test, test_sizes, test_hashes = _scan(paths["test"], "test", problems)

    if not train:
        problems.append("The training archive has no images.")
    if not test:
        problems.append("The hidden test archive has no images.")
    if set(train) != set(test):
        problems.append(f"Train and test classes differ: only in train {sorted(set(train) - set(test))}, "
                        f"only in test {sorted(set(test) - set(train))}")
    shared = train_hashes & test_hashes
    if shared:
        problems.append(f"{len(shared)} test images are exact copies of training images.")

    sizes = train_sizes + test_sizes
    archive_mb = {k: os.path.getsize(p) / (1024 * 1024) for k, p in paths.items()}
    total_mb = sum(archive_mb.values())
    if total_mb > SIZE_WARNING_MB:
        warnings.append(f"Image archives total {total_mb:.1f} MB (platform guideline is under "
                        f"{SIZE_WARNING_MB} MB).")

    summary = {
        "classes": sorted(train),
        "train_per_class": dict(sorted(train.items())),
        "test_per_class": dict(sorted(test.items())),
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
    return (
        f"Classes ({len(summary['classes'])}): {', '.join(summary['classes'])}\n"
        f"Train images per class: {summary['train_per_class']}\n"
        f"Hidden test images per class: {summary['test_per_class']}\n"
        f"Image sizes: {summary['size_range']}\n"
        f"Archives: training {summary['train_mb']} MB, hidden test {summary['test_mb']} MB"
    )


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
