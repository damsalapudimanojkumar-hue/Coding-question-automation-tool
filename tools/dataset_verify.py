"""Check a suggested image dataset BEFORE it is shown, without downloading the images.

A model suggests datasets from memory, so a suggestion can be wrong: the slug may not
exist, the dataset may hold no labels, its classes may differ, or it may be huge. This
module looks at the real thing and reports verified facts:
- Kaggle: the file summary (image count, size) and the folder tree (class folders and
  their image counts) come from Kaggle's API in about a second; small label tables are
  downloaded on their own (a few seconds) and read with the same rules the builder uses.
- torchvision: the dataset class must exist and be downloadable.

Each result is compared with the objective's data requirement (labels needed or not), so
an unsuitable dataset is filtered out with a reason, and the reason is fed back to the
next round of suggestions.
"""

from __future__ import annotations

import os
import posixpath
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from tools.image_layout import IMAGE_EXTS
from tools.image_sources import (DOWNLOAD_ROOT, NOT_CLASS_NAMES, SPLIT_NAMES, _norm, _tv_class,
                                 pick_label_column, read_table)

MAX_LISTINGS = 30            # folder listings per dataset (each about 0.7 s)
MAX_TABLES = 3               # label tables downloaded per dataset
MAX_TABLE_MB = 20
WARN_GB, REJECT_GB = 2, 20
FILE_COLUMN_HINT = re.compile(r"file|path|image|img|(^|[_\s])id$|name", re.I)

_CLIENT = None


def _api():
    global _CLIENT
    if _CLIENT is None:
        from kagglesdk import KaggleClient
        token = os.environ.get("KAGGLE_API_TOKEN")
        if token:
            _CLIENT = KaggleClient(api_token=token)
        else:
            _CLIENT = KaggleClient(username=os.environ.get("KAGGLE_USERNAME"),
                                   password=os.environ.get("KAGGLE_KEY"))
    return _CLIENT.datasets.dataset_api_client


_KAGGLE_OK = None


def _kaggle_works() -> bool:
    """Kaggle answers 403 both for a missing dataset and for bad credentials; asking about a
    well-known public dataset tells the two apart (checked once per process)."""
    global _KAGGLE_OK
    if _KAGGLE_OK is None:
        try:
            _summary("alxmamaev", "flowers-recognition")
            _KAGGLE_OK = True
        except Exception:  # noqa: BLE001
            _KAGGLE_OK = False
    return _KAGGLE_OK


def _summary(owner: str, slug: str):
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetFilesSummaryRequest
    req = ApiGetDatasetFilesSummaryRequest()
    req.owner_slug, req.dataset_slug = owner, slug
    return _api().get_dataset_files_summary(req)


def _listing(owner: str, slug: str, path: str):
    from kagglesdk.datasets.types.dataset_api_service import ApiListTreeDatasetFilesRequest
    req = ApiListTreeDatasetFilesRequest()
    req.owner_slug, req.dataset_slug, req.page_size = owner, slug, 1000
    if path:
        req.path = path
    return _api().list_tree_dataset_files(req)


def _walk_tree(owner: str, slug: str) -> tuple[list, list]:
    """(class-folder groups, tables). A group is a folder whose sub-folders hold files and
    are not generic names (train/, images/); tables are (path, bytes) of .csv files."""
    groups, tables = [], []
    queue, listings = [""], 0
    while queue and listings < MAX_LISTINGS:
        path = queue.pop(0)
        listing = _listing(owner, slug, path)
        listings += 1
        for f in listing.files or []:
            if f.name.lower().endswith(".csv"):
                tables.append((f.relative_url or posixpath.join(path, f.name), f.total_bytes or 0))
        leaves = {}
        for d in listing.directories or []:
            child = d.relative_url or posixpath.join(path, d.name)
            if d.total_directories:
                if path.count("/") < 5:
                    queue.append(child)
            elif d.total_files and d.name.lower() not in NOT_CLASS_NAMES and not d.name.startswith("."):
                leaves[d.name] = d.total_files
        if len(leaves) >= 2:
            split = next((SPLIT_NAMES[p.lower()] for p in path.split("/") if p.lower() in SPLIT_NAMES), None)
            groups.append({"path": path, "classes": leaves, "split": split})
    return groups, tables


def _read_label_table(slug: str, rel: str) -> dict:
    """Download one table on its own and find its image column and label column."""
    import kagglehub
    out = os.path.join(DOWNLOAD_ROOT, "verify", slug.replace("/", "__"))
    local = kagglehub.dataset_download(slug, path=rel, output_dir=out)
    header, rows = read_table(local)
    if len(rows) < 4:
        return {"path": rel, "reason": "too few rows"}
    file_j, best = None, 0
    for j, name in enumerate(header):
        values = [r[j].strip() for r in rows[:500]]
        looks = sum(v.lower().endswith(IMAGE_EXTS) for v in values)
        unique = len(set(values)) >= 0.9 * len(values)
        score = looks + (len(values) // 2 if unique and FILE_COLUMN_HINT.search(name) else 0)
        if score > best:
            file_j, best = j, score
    if file_j is None or best < 0.5 * min(len(rows), 500):
        return {"path": rel, "reason": "no column names the image files"}
    label_j, _ = pick_label_column(header, rows, {file_j})
    if label_j is None:
        return {"path": rel, "reason": f"lists images in '{header[file_j]}' but has no label column"}
    per_image = {}
    for r in rows:
        per_image.setdefault(r[file_j].strip(), set()).add(r[label_j].strip())
    if sum(len(v) > 1 for v in per_image.values()) > 0.01 * len(per_image):
        return {"path": rel, "reason": "several labels per image (boxes or multi-label)"}
    counts = Counter(r[label_j].strip() for r in rows)
    return {"path": rel, "file_column": header[file_j], "label_column": header[label_j],
            "classes": dict(counts)}


def _check_kaggle(slug: str) -> dict:
    owner, _, name = slug.partition("/")
    summary = _summary(owner, name).file_summary_info
    images = [t for t in (summary.file_types or []) if (t.extension or "").lower() in IMAGE_EXTS]
    n_images = sum(t.file_count or 0 for t in images)
    size_gb = sum(t.total_size or 0 for t in (summary.file_types or [])) / 1024 ** 3
    facts = {"images": n_images, "size_gb": size_gb,
             "formats": sorted({t.extension.lower() for t in images}), "label_sources": [],
             "not_labels": [], "classes": {}, "official_test_split": False}
    if not n_images:
        return facts
    groups, tables = _walk_tree(owner, name)
    if groups:
        merged = Counter()
        for g in groups:
            merged.update(g["classes"])
        facts["label_sources"].append(f"class folders ({len(merged)} classes)")
        facts["classes"] = dict(merged)
        facts["official_test_split"] = any(g["split"] == "test" for g in groups)
    candidates = sorted((t for t in tables if "submission" not in t[0].lower()
                         and t[1] <= MAX_TABLE_MB * 1048576), key=lambda t: -t[1])[:MAX_TABLES]
    for rel, _ in candidates:
        try:
            table = _read_label_table(slug, rel)
        except Exception as exc:  # noqa: BLE001 - one unreadable table is just not a label source
            facts["not_labels"].append(f"{rel}: could not be read ({type(exc).__name__})")
            continue
        if "reason" in table:
            facts["not_labels"].append(f"{rel}: {table['reason']}")
            continue
        facts["label_sources"].append(f"{rel} (labels in '{table['label_column']}', "
                                      f"{len(table['classes'])} classes)")
        if not facts["classes"]:
            facts["classes"] = table["classes"]
    return facts


def _check_torchvision(source_id: str) -> dict:
    import inspect
    cls = _tv_class(source_id)
    if "download" not in inspect.signature(cls.__init__).parameters:
        raise LookupError(f"torchvision.{cls.__name__} cannot be downloaded automatically")
    return {"images": None, "size_gb": None, "formats": [], "classes": {}, "not_labels": [],
            "label_sources": ["torchvision labels (written as class folders)"], "official_test_split": False}


def verify_option(option: dict, requirement: dict) -> dict:
    """{"status": ok|warn|reject|unverified, "facts": [...], "reasons": [...]} for one option."""
    source, source_id = option.get("source"), option.get("source_id") or ""
    try:
        if source == "kaggle":
            if source_id.count("/") != 1:
                return _result("reject", [], [f"'{source_id}' is not an owner/dataset Kaggle slug"])
            facts = _check_kaggle(source_id)
        elif source == "torchvision":
            facts = _check_torchvision(source_id)
        else:
            return _result("reject", [], [f"unknown source '{source}'"])
    except Exception as exc:  # noqa: BLE001 - classify: missing dataset vs. a check that could not run
        text = str(exc)
        if any(s in text for s in ("has no dataset", "cannot be downloaded")):
            return _result("reject", [], [f"not available: {text[:160]}"])
        if any(s in text for s in ("403", "404", "Not Found")) and _kaggle_works():
            return _result("reject", [], ["not found on Kaggle (wrong slug, removed, or private)"])
        return _result("unverified", [], [f"could not be checked ({type(exc).__name__}: {text[:120]})"])

    lines, reasons, status = [], [], "ok"
    if facts["images"] is not None:
        if not facts["images"]:
            return _result("reject", [], ["the dataset contains no image files"])
        lines.append(f"{facts['images']:,} images ({', '.join(facts['formats'])}), "
                     f"{facts['size_gb']:.1f} GB download")
    lines += [f"Labels: {s}" for s in facts["label_sources"]] or ["Labels: none found (images only)"]
    lines += [f"Not labels: {s}" for s in facts["not_labels"]]
    if facts["official_test_split"]:
        lines.append("Has its own test split")

    needs_labels = requirement.get("labels") == "required"
    if needs_labels and not facts["label_sources"]:
        return _result("reject", lines, ["the objective needs labelled images, and this dataset has no labels"])
    if facts["size_gb"] and facts["size_gb"] > REJECT_GB:
        return _result("reject", lines, [f"too large to download ({facts['size_gb']:.0f} GB)"])
    if facts["size_gb"] and facts["size_gb"] > WARN_GB:
        status = "warn"
        reasons.append(f"large download ({facts['size_gb']:.1f} GB): the first use takes a while")
    suggested = option.get("classes") or []
    if facts["classes"] and suggested:
        found = {_norm(c) for c in facts["classes"]}
        missing = [c for c in suggested if _norm(c) not in found]
        if missing:
            status = "warn"
            real = sorted(facts["classes"], key=facts["classes"].get, reverse=True)
            reasons.append(f"suggested classes not in the dataset: {', '.join(map(str, missing))}; "
                           f"its classes include {', '.join(map(str, real[:12]))}")
    return _result(status, lines, reasons)


def _result(status: str, facts: list, reasons: list) -> dict:
    return {"status": status, "facts": facts, "reasons": reasons}


def verify_options(options: list, requirement: dict) -> list:
    """Verify several options in parallel; each option gets a "verified" entry."""
    with ThreadPoolExecutor(max_workers=max(1, min(5, len(options)))) as pool:
        results = list(pool.map(lambda o: verify_option(o, requirement), options))
    for option, result in zip(options, results):
        option["verified"] = result
    return options
