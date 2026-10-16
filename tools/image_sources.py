"""Fetch, inspect and sample real image datasets for CV assignments.

Fixed, tested code instead of model-written code: the model only suggests which
dataset to use; downloading, reading labels, sampling and zipping happen here, the
same way every run.

Datasets are used the way they come. Nothing is renamed, restructured or invented:
- Images keep their original file names, folders, format and size. Images a dataset
  only holds in memory (e.g. torchvision's SVHN, CIFAR) are saved losslessly as PNG.
- Labels are read from wherever the dataset keeps them, and every place is offered:
    folders  one folder per class (folder name = label)
    csv      the dataset's own label table(s): a column naming each image and a column
             with its class. A sample is the same table with fewer rows.
    none     no labels: the images alone (for objectives that do not need labels)
- Students get the sampled images plus the trimmed label table, if any; the hidden
  grading images go only into tests/, with their own trimmed rows.

Sources:
    torchvision  a class in torchvision.datasets (all of its splits are loaded)
    kaggle       an "owner/dataset" slug, in whatever layout it is published

Leakage rules (the hidden grading set must never contain what students train on):
- When a dataset ships its own test split, hidden images come from that split and
  student images from the rest, as long as every chosen class has enough of both.
  Otherwise images are pooled and split at random.
- Byte-identical images are used at most once across all classes and both archives.
- A Kaggle download holding several variants of the same photos (e.g. color/,
  grayscale/, segmented/ copies of one class tree) uses only one variant.
"""

from __future__ import annotations

import csv
import hashlib
import inspect as _inspect
import io
import os
import posixpath
import random
import re
import tempfile
import zipfile
from collections import Counter

from PIL import Image

from naming import images_zip_name, test_images_zip_relname
from tools.image_layout import IMAGE_EXTS, class_order

ALREADY_COMPRESSED = (".jpg", ".jpeg", ".png", ".webp")
DOWNLOAD_ROOT = os.path.join(tempfile.gettempdir(), "assignment_pipeline_datasets")
SPLIT_NAMES = {"train": "train", "training": "train", "val": "train", "valid": "train",
               "validation": "train", "trainval": "train", "test": "test", "testing": "test"}
NOT_CLASS_NAMES = {"images", "imgs", "data", "dataset", "raw", "raw-img"} | set(SPLIT_NAMES)
UNLABELLED = "(no label)"
MAX_TABLE_MB = 50
# Column names that usually hold the class (used only to rank otherwise equal candidates).
LABEL_HINT = re.compile(r"label|class|categor|target|species|breed|type|diagnos|disease|"
                        r"emotion|grade|sport|genre|name", re.I)
NOT_LABEL_HINT = re.compile(r"file|path|image|img|(^|[_\s])id$|url|date|time", re.I)

_HANDLES: dict = {}   # (source, source_id) -> loaded dataset, reused between inspect and build


class DatasetUnavailable(Exception):
    """The dataset could not be downloaded or holds nothing usable."""


def _folder_mb(path: str) -> float:
    total = 0
    for folder, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(folder, f))
            except OSError:
                pass
    return total / (1024 * 1024)


def _safe_name(name) -> str:
    """Folder-safe class name; names with no ASCII letters get a stable unique fallback."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_")
    return safe or "class_" + hashlib.md5(str(name).encode("utf-8")).hexdigest()[:6]


def _norm(name) -> str:
    """Loose form for matching a suggested class name to the dataset's own spelling."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _split_of(rel: str):
    """'train'/'test' when a folder on the image's path names a split, else None."""
    return next((SPLIT_NAMES[p.lower()] for p in rel.split("/")[:-1] if p.lower() in SPLIT_NAMES), None)


# ── torchvision ────────────────────────────────────────────────────────────
def _tv_class(source_id: str):
    import torchvision.datasets as tvd
    cls = getattr(tvd, source_id, None)
    if cls is None:
        match = [n for n in dir(tvd) if n.lower() == source_id.lower()]
        cls = getattr(tvd, match[0]) if match else None
    if cls is None:
        raise DatasetUnavailable(f"torchvision has no dataset named '{source_id}'.")
    return cls


def _tv_parts(cls, root: str) -> list:
    """[(split_tag, dataset)] for every split the class offers; split_tag is "train",
    "test" or None (no official split). An unknown split name is skipped; any other
    failure (download, corrupt files) stops the load instead of keeping a partial dataset."""
    params = _inspect.signature(cls.__init__).parameters
    kwargs = {"root": root, "download": True}
    if "split" in params:
        attempts = [("train", {"split": "train"}), ("train", {"split": "val"}),
                    ("test", {"split": "test"})]
    elif "train" in params:
        attempts = [("train", {"train": True}), ("test", {"train": False})]
    else:
        attempts = [(None, {})]
    parts = []
    for tag, extra in attempts:
        try:
            parts.append((tag, cls(**extra, **kwargs)))
        except ValueError:            # torchvision's verify_str_arg: this split does not exist
            continue
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, never cached half-loaded
            raise DatasetUnavailable(f"Could not load {cls.__name__} {extra or ''}: {exc}") from exc
    if "split" in params and not any(tag == "train" for tag, _ in parts):
        try:
            parts.append(("train", cls(split="trainval", **kwargs)))   # e.g. OxfordIIITPet
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001
            raise DatasetUnavailable(f"Could not load {cls.__name__} (trainval): {exc}") from exc
    if not parts:
        raise DatasetUnavailable(f"Could not load {cls.__name__}: no usable split.")
    return parts


def _tv_labels(ds) -> list[int]:
    for attr in ("targets", "_labels", "labels", "y"):
        values = getattr(ds, attr, None)
        if values is not None and len(values) == len(ds):
            return [int(v) for v in values]
    samples = getattr(ds, "_samples", None) or getattr(ds, "samples", None)
    if samples is not None and len(samples) == len(ds):
        return [int(s[1]) for s in samples]
    raise DatasetUnavailable(f"Cannot read the labels of {type(ds).__name__} without decoding every image.")


def _is_path(value) -> bool:
    return isinstance(value, (str, os.PathLike))


def _tv_paths(ds):
    """Original image files behind the dataset, in index order, when it has them."""
    for attr in ("samples", "imgs", "_samples"):
        pairs = getattr(ds, attr, None)
        if pairs is not None and len(pairs) == len(ds) and len(ds) and _is_path(pairs[0][0]):
            return [str(p[0]) for p in pairs]
    for attr in ("_image_files", "_images", "images", "data"):
        files = getattr(ds, attr, None)
        if isinstance(files, (list, tuple)) and len(files) == len(ds) and len(ds) and _is_path(files[0]):
            return [str(f) for f in files]
    return None


def _load_torchvision(source_id: str) -> dict:
    """torchvision keeps labels inside the dataset object; they are written out as class
    folders (train/<class>/, test/<class>/), which is how torchvision itself names them."""
    cls = _tv_class(source_id)
    root = os.path.join(DOWNLOAD_ROOT, "torchvision")
    os.makedirs(root, exist_ok=True)
    before = _folder_mb(root)
    parts = _tv_parts(cls, root)
    grown = _folder_mb(root) - before
    items: dict[str, list] = {}
    for p, (tag, ds) in enumerate(parts):
        names = getattr(ds, "classes", None) or getattr(ds, "categories", None)
        paths = _tv_paths(ds)
        for i, label in enumerate(_tv_labels(ds)):
            if label < 0:
                continue          # unlabeled samples (e.g. STL10 'unlabeled')
            name = names[label] if names and label < len(names) else str(label)
            items.setdefault(_safe_name(name), []).append((paths[i] if paths else (p, i), tag))
    labelled = {"key": "folders", "kind": "folders", "items": items,
                "title": "Class labels from torchvision, written as one folder per class"}
    unlabelled = {"key": "none", "kind": "none", "title": "No labels (images only)",
                  "items": {UNLABELLED: [ref for refs in items.values() for ref in refs]}}
    return {"name": cls.__name__, "root": None, "parts": [ds for _, ds in parts],
            "sources": {"folders": labelled, "none": unlabelled}, "unused_tables": [], "variants": [],
            "variant_used": "",
            "download": f"downloaded {grown:.0f} MB" if grown > 1 else "already downloaded"}


# ── kaggle: find every place the labels can come from ─────────────────────
def _kaggle_files(root: str) -> tuple[list, list]:
    """(image rel paths, table rel paths) in the download, skipping hidden and macOS files."""
    images, tables = [], []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__MACOSX")
        for f in sorted(files):
            if f.startswith("."):
                continue
            rel = os.path.relpath(os.path.join(folder, f), root).replace(os.sep, "/")
            low = f.lower()
            if low.endswith(IMAGE_EXTS):
                images.append(rel)
            elif low.endswith(".csv") and os.path.getsize(os.path.join(folder, f)) <= MAX_TABLE_MB * 1048576:
                tables.append(rel)
    return images, tables


def _folder_source(root: str, images: list):
    """Labels from class folders: an image's parent folder is its class, unless that folder
    is a generic name (train/, images/, ...). Several variants of one class tree (color/,
    grayscale/) are never pooled; the largest is used."""
    variants: dict[str, dict] = {}
    seen = set()
    for rel in images:
        parts = rel.split("/")
        if len(parts) < 2 or parts[-2].lower() in NOT_CLASS_NAMES:
            continue
        cls = parts[-2]
        variant = "/".join(p for p in parts[:-2] if p.lower() not in SPLIT_NAMES) or "."
        full = os.path.join(root, *parts)
        key = (variant, cls, parts[-1], os.path.getsize(full))     # duplicate copies of one file
        if key not in seen:
            seen.add(key)
            variants.setdefault(variant, {}).setdefault(cls, []).append((full, _split_of(rel)))
    if not variants:
        return None
    size = {v: sum(len(refs) for refs in classes.values()) for v, classes in variants.items()}
    chosen = max(sorted(variants), key=size.get)
    return {"key": "folders", "kind": "folders", "items": variants[chosen],
            "title": "Class folders (each folder name is the label)",
            "variants": sorted(variants) if len(variants) > 1 else [], "variant_used": chosen}


def read_table(path: str) -> tuple[list, list]:
    """(header, rows) of a CSV file; rows with the wrong number of cells are dropped."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with open(path, newline="", encoding=encoding) as fh:
                rows = list(csv.reader(fh))
            break
        except UnicodeDecodeError:
            continue
    if not rows:
        return [], []
    header = [h.strip() for h in rows[0]]
    return header, [r for r in rows[1:] if len(r) == len(header)]


def pick_label_column(header: list, rows: list, skip: set) -> tuple:
    """(label column index or None, split column index or None) for a table whose rows are
    images. A label column has 2 to 1000 distinct values, repeats across rows, and is not
    another path or id column; column names only break ties between candidates."""
    best, split_j = None, None
    n = len(rows)
    for j, name in enumerate(header):
        if j in skip:
            continue
        values = [r[j].strip() for r in rows]
        filled = [v for v in values if v]
        if len(filled) < 0.95 * n:
            continue
        distinct = set(filled)
        if {v.lower() for v in distinct} <= set(SPLIT_NAMES):
            split_j = j
            continue
        if len(distinct) < 2 or len(distinct) > min(1000, max(2, n // 2)):
            continue
        if any(v.lower().endswith(IMAGE_EXTS) for v in list(distinct)[:50]):
            continue
        numeric = all(v.lstrip("-").replace(".", "", 1).isdigit() for v in list(distinct)[:200])
        score = (10 if LABEL_HINT.search(name) else 0) - (10 if NOT_LABEL_HINT.search(name) else 0) \
            + (0 if numeric else 2)
        if best is None or score > best[0]:
            best = (score, j)
    return (best[1] if best else None), split_j


def _resolver(table_rel: str, values: list, image_set: set, by_name: dict, exts: list):
    """Find how a column's values point at image files: (base, suffix, hits). base is the
    folder the values are relative to ("" = dataset root, "*" = look the name up anywhere);
    suffix is an extension the values leave out."""
    table_dir = posixpath.dirname(table_rel)
    best = (None, "", 0)
    for suffix in [""] + exts:
        for base in dict.fromkeys([table_dir, "", "*"]):
            hits = 0
            for v in values:
                name = v.strip().replace("\\", "/") + suffix
                if base == "*":
                    found = len(by_name.get(name.split("/")[-1], ())) == 1
                else:
                    found = posixpath.normpath(posixpath.join(base, name)) in image_set
                hits += found
            if hits > best[2]:
                best = (base, suffix, hits)
    return best


def _resolve(base: str, suffix: str, value: str, image_set: set, by_name: dict):
    name = value.strip().replace("\\", "/") + suffix
    if base == "*":
        found = by_name.get(name.split("/")[-1], ())
        return found[0] if len(found) == 1 else None
    rel = posixpath.normpath(posixpath.join(base, name))
    return rel if rel in image_set else None


def _table_sources(root: str, images: list, tables: list) -> tuple[dict, list]:
    """Label tables shipped with the dataset, grouped by (image column, label column).
    Returns ({key: source}, [unused tables with the reason])."""
    image_set = set(images)
    by_name: dict = {}
    for rel in images:
        by_name.setdefault(rel.split("/")[-1], []).append(rel)
    exts = [e for e, _ in Counter(posixpath.splitext(r)[1].lower() for r in images).most_common(3)]
    exts = list(dict.fromkeys(exts + [e.upper() for e in exts]))
    parsed, unused = [], []
    for rel in tables:
        if "submission" in rel.lower():
            unused.append({"path": rel, "reason": "an answer template for competitions, not labels"})
            continue
        header, rows = read_table(os.path.join(root, *rel.split("/")))
        if len(rows) < 4:
            unused.append({"path": rel, "reason": "too few rows"})
            continue
        sample = rows if len(rows) <= 300 else random.Random(0).sample(rows, 300)
        best = None
        for j in range(len(header)):
            base, suffix, hits = _resolver(rel, [r[j] for r in sample], image_set, by_name, exts)
            if hits >= max(2, 0.8 * len(sample)) and (best is None or hits > best[3]):
                best = (j, base, suffix, hits)
        if best is None:
            unused.append({"path": rel, "reason": "no column names the image files"})
            continue
        file_j, base, suffix, _ = best
        resolved = [(_resolve(base, suffix, r[file_j], image_set, by_name), i, r) for i, r in enumerate(rows)]
        resolved = [x for x in resolved if x[0]]
        label_j, split_j = pick_label_column(header, [r for _, _, r in resolved], {file_j})
        if label_j is None:
            unused.append({"path": rel, "reason": f"lists images in '{header[file_j]}' but has no label column"})
            continue
        labels: dict = {}
        for img, _, r in resolved:
            labels.setdefault(img, set()).add(r[label_j].strip())
        conflicting = sum(len(v) > 1 for v in labels.values())
        if conflicting > 0.01 * len(labels):
            unused.append({"path": rel, "reason": f"{conflicting} images have several labels each "
                                                  f"(boxes or multi-label), not one class per image"})
            continue
        if base == "*":       # bare file names: name the folder when they all live in one
            dirs = {posixpath.dirname(img) for img, _, _ in resolved}
            if len(dirs) == 1 and all("/" not in r[file_j] for _, _, r in resolved):
                base = dirs.pop()
        stem_split = next((SPLIT_NAMES[t] for t in re.split(r"[^a-z]+", posixpath.basename(rel).lower())
                           if t in SPLIT_NAMES), None)
        names_only = all("/" not in r[file_j].replace("\\", "/") for _, _, r in resolved)
        parsed.append({"path": rel, "header": header, "rows": rows, "file_j": file_j, "label_j": label_j,
                       "split_j": split_j, "base": base, "suffix": suffix, "resolved": resolved,
                       "names_only": names_only,
                       "conflicting": {img for img, v in labels.items() if len(v) > 1},
                       "stem_split": stem_split})

    sources: dict = {}
    for t in sorted(parsed, key=lambda t: -len(t["resolved"])):   # biggest table wins a shared image
        file_col, label_col = t["header"][t["file_j"]], t["header"][t["label_j"]]
        key = f"csv:{file_col}->{label_col}"
        src = sources.setdefault(key, {"key": key, "kind": "csv", "items": {}, "rows": {}, "tables": {},
                                       "file_column": file_col, "label_column": label_col})
        src["tables"][t["path"]] = {k: t[k] for k in ("header", "rows", "base", "suffix", "names_only")}
        for img, i, r in t["resolved"]:
            full = os.path.join(root, *img.split("/"))
            if img in t["conflicting"] or full in src["rows"]:
                continue
            tag = None
            if t["split_j"] is not None:
                tag = SPLIT_NAMES.get(r[t["split_j"]].strip().lower())
            tag = tag or t["stem_split"] or _split_of(img)
            src["items"].setdefault(r[t["label_j"]].strip(), []).append((full, tag))
            src["rows"][full] = (t["path"], i)
    for src in sources.values():
        names = list(src["tables"])
        src["title"] = (f"Label table{'s' if len(names) > 1 else ''} {', '.join(names)} "
                        f"(images in '{src['file_column']}', labels in '{src['label_column']}')")
    return sources, unused


def _load_kaggle(slug: str) -> dict:
    import kagglehub
    try:
        path = kagglehub.dataset_download(slug)
    except Exception as exc:  # noqa: BLE001 - auth/network/not-found all mean "unavailable"
        raise DatasetUnavailable(f"Kaggle download failed for '{slug}': {exc}") from exc
    root = os.path.normpath(path)
    images, tables = _kaggle_files(root)
    if not images:
        raise DatasetUnavailable(f"'{slug}' contains no image files.")
    sources = {}
    folders = _folder_source(root, images)
    if folders:
        sources["folders"] = folders
    table_sources, unused = _table_sources(root, images, tables)
    sources.update(table_sources)
    if folders:            # unlabelled use of a class-folder dataset: one variant, no pooling
        pool = [ref for refs in folders["items"].values() for ref in refs]
    else:
        pool = [(os.path.join(root, *rel.split("/")), _split_of(rel)) for rel in images]
    sources["none"] = {"key": "none", "kind": "none", "items": {UNLABELLED: pool},
                       "title": "No labels (images only)"}
    return {"name": slug.split("/")[-1], "root": root, "parts": None, "sources": sources,
            "unused_tables": unused,
            "variants": folders["variants"] if folders else [],
            "variant_used": folders["variant_used"] if folders else "",
            "download": f"{_folder_mb(path):.0f} MB on disk"}


# ── public API ─────────────────────────────────────────────────────────────
def _handle(source: str, source_id: str) -> dict:
    key = (source, source_id)
    if key not in _HANDLES:
        if source == "torchvision":
            handle = _load_torchvision(source_id)
        elif source == "kaggle":
            handle = _load_kaggle(source_id)
        else:
            raise DatasetUnavailable(f"Unknown source '{source}' (expected torchvision or kaggle).")
        usable = {}
        for k, src in handle["sources"].items():
            # A class needs at least one training and one hidden test image.
            src["items"] = {c: refs for c, refs in src["items"].items() if len(refs) >= 2}
            if len(src["items"]) >= (1 if src["kind"] == "none" else 2):
                usable[k] = src
        if not usable:
            raise DatasetUnavailable(f"'{source_id}' has no usable images.")
        handle["sources"] = usable
        _HANDLES[key] = handle
    return _HANDLES[key]


def _payload(handle: dict, ref) -> tuple[bytes, str]:
    """(file bytes, extension) for one image at its natural size; raises if unreadable."""
    if isinstance(ref, str):
        with Image.open(ref) as img:
            img.load()                         # full decode: truncated files are rejected
        with open(ref, "rb") as fh:
            return fh.read(), os.path.splitext(ref)[1].lower()
    part, index = ref
    image = handle["parts"][part][index][0]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    buf = io.BytesIO()
    image.save(buf, "PNG")                     # lossless, original size
    return buf.getvalue(), ".png"


def _source_info(src: dict) -> dict:
    return {
        "key": src["key"], "kind": src["kind"], "title": src["title"],
        "labelled": src["kind"] != "none",
        "classes": [{"name": k, "count": len(v)} for k, v in sorted(src["items"].items())],
        "total": sum(len(v) for v in src["items"].values()),
        "official_test_split": any(tag == "test" for refs in src["items"].values() for _, tag in refs),
    }


def inspect_dataset(source: str, source_id: str) -> dict:
    """Download (or reuse) the dataset and report what is really in it, including every
    place its labels can come from."""
    handle = _handle(source, source_id)
    sources = [_source_info(s) for s in handle["sources"].values()]
    labelled = [s for s in sources if s["labelled"]]
    default = max(labelled, key=lambda s: (s["total"], s["kind"] == "folders"))["key"] if labelled \
        else sources[0]["key"]
    rng = random.Random(0)
    pool = list({ref for s in handle["sources"].values() for refs in s["items"].values() for ref, _ in refs})
    pool.sort(key=str)
    sizes, kbs = [], []
    for ref in rng.sample(pool, min(20, len(pool))):
        try:
            data, _ = _payload(handle, ref)
            with Image.open(io.BytesIO(data)) as img:
                sizes.append(img.size)
            kbs.append(len(data) / 1024)
        except Exception:  # noqa: BLE001 - a broken sample file just isn't measured
            continue
    if sizes:
        smallest = min(sizes, key=lambda s: s[0] * s[1])
        largest = max(sizes, key=lambda s: s[0] * s[1])
        size_range = (f"{smallest[0]}x{smallest[1]}" if smallest == largest
                      else f"{smallest[0]}x{smallest[1]} to {largest[0]}x{largest[1]}")
    else:
        size_range = "unknown"
    return {
        "source": source,
        "source_id": source_id,
        "name": handle["name"],
        "download": handle["download"],
        "images": len(pool),
        "size_range": size_range,
        "avg_kb": round(sum(kbs) / len(kbs), 1) if kbs else 0.0,
        "label_sources": sources,
        "default_source": default,
        "unused_tables": handle.get("unused_tables", []),
        "variants": handle.get("variants", []),
        "variant_used": handle.get("variant_used", ""),
    }


def source_by_key(info: dict, key: str) -> dict:
    return next((s for s in info["label_sources"] if s["key"] == key), info["label_sources"][0])


def default_choice(info: dict, suggested: list, use_labels: bool = True) -> dict:
    """Default label source (labelled when the objective needs labels, else images only),
    the suggested classes that really exist (else the 10 largest), up to 150 images each."""
    key = info["default_source"]
    if not use_labels and any(s["kind"] == "none" for s in info["label_sources"]):
        key = "none"
    src = source_by_key(info, key)
    counts = {c["name"]: c["count"] for c in src["classes"]}
    if not src["labelled"]:
        return {"label_source": key, "classes": [UNLABELLED],
                "images_per_class": min(1000, counts[UNLABELLED]), "use_all": False, "test_fraction": 0.2}
    wanted = {_norm(s) for s in suggested or []}
    classes = [name for name in counts if _norm(name) in wanted]
    if len(classes) < 2:
        classes = sorted(counts, key=counts.get, reverse=True)[:10]
    return {"label_source": key, "classes": sorted(classes),
            "images_per_class": min(150, min(counts[c] for c in classes)),
            "use_all": False, "test_fraction": 0.2}


def _test_count(n: int, test_fraction: float) -> int:
    return min(n - 1, max(1, round(n * test_fraction)))


def estimate(class_counts: list, per_class, test_fraction: float, avg_kb: float) -> dict:
    """Totals for the choice panel. per_class=None means every image of each class."""
    used = [c if per_class is None else min(per_class, c) for c in class_counts]
    test = sum(_test_count(n, test_fraction) for n in used)
    total = sum(used)
    return {"total": total, "train": total - test, "test": test,
            "zip_mb": total * avg_kb / 1024}


def _pools(src, cls) -> tuple[list, list, list]:
    """(official train refs, official test refs, all refs), each in a stable order."""
    entries = sorted(src["items"][cls], key=lambda e: str(e[0]))
    return ([r for r, tag in entries if tag != "test"], [r for r, tag in entries if tag == "test"],
            [r for r, _ in entries])


def _plan(handle, src, choice) -> tuple[dict, bool]:
    """Pick the student and hidden images of every class before anything is written."""
    classes = list(choice["classes"])
    missing = [c for c in classes if c not in src["items"]]
    if missing:
        raise DatasetUnavailable(f"These classes are not in the chosen labels: {missing}")
    per_class = None if choice.get("use_all") else int(choice["images_per_class"])
    fraction = float(choice["test_fraction"])
    wants = {}
    for cls in classes:
        train_refs, test_refs, all_refs = _pools(src, cls)
        total = len(all_refs) if per_class is None else per_class
        n_test = _test_count(total, fraction)
        wants[cls] = (train_refs, test_refs, all_refs, total - n_test, n_test)
    official = all(test and train and (per_class is None or (len(train) >= n_train and len(test) >= n_test))
                   for train, test, _, n_train, n_test in wants.values())
    rng = random.Random(42)
    seen: set = set()                           # content hashes already used anywhere
    plan = {}
    for cls in classes:
        train_refs, test_refs, all_refs, n_train, n_test = wants[cls]
        if official:
            hidden = _take(handle, test_refs, None if per_class is None else n_test, rng, seen)
            student = _take(handle, train_refs, None if per_class is None else n_train, rng, seen)
        else:
            picked = _take(handle, all_refs, None if per_class is None else n_train + n_test, rng, seen)
            n_hidden = _test_count(len(picked), fraction) if len(picked) >= 2 else 0
            hidden, student = picked[:n_hidden], picked[n_hidden:]
        if not hidden or not student or (per_class is not None and len(hidden) + len(student) < per_class):
            what = "the dataset" if cls == UNLABELLED else f"Class '{cls}'"
            raise DatasetUnavailable(
                f"{what} has only {len(student) + len(hidden)} usable images "
                f"(after skipping unreadable and duplicate files); asked for "
                f"{'every image' if per_class is None else per_class}.")
        plan[cls] = (student, hidden)
    return plan, official


def _trimmed_table(table: dict, keep_rows: list) -> bytes:
    """The dataset's own table with only the given rows, columns and order unchanged."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(table["header"])
    for i in sorted(keep_rows):
        writer.writerow(table["rows"][i])
    return buf.getvalue().encode("utf-8")


def _spec(src, entries: list, tables: list) -> dict:
    """Layout spec for one archive, from what was actually written into it."""
    if src["kind"] == "folders":
        return {"kind": "folders",
                "roots": sorted({posixpath.dirname(posixpath.dirname(e)) for e in entries})}
    if src["kind"] == "csv":
        return {"kind": "csv", "tables": [
            {"path": path, "columns": src["tables"][path]["header"], "file_column": src["file_column"],
             "label_column": src["label_column"], "base": src["tables"][path]["base"],
             "suffix": src["tables"][path]["suffix"],
             "names_only": src["tables"][path]["names_only"]} for path in tables]}
    return {"kind": "none", "dirs": sorted({posixpath.dirname(e) for e in entries})}


def build_archives(source: str, source_id: str, choice: dict, workspace: str, code: str) -> dict:
    """Sample and zip: images_<CODE>.zip (students) and tests/test_images_<CODE>.zip (hidden).
    Kaggle images keep their original paths and names; a label table is copied with only the
    rows of the images in that archive. Unreadable and duplicate files are skipped.
    Returns {"official_split": bool, "layout": {...}} (see tools.image_layout)."""
    handle = _handle(source, source_id)
    sources = handle["sources"]
    src = sources.get(choice.get("label_source") or "") or sources.get("folders") or next(iter(sources.values()))
    plan, official = _plan(handle, src, choice)

    targets = {"train": os.path.join(workspace, images_zip_name(code)),
               "test": os.path.join(workspace, *test_images_zip_relname(code).split("/"))}
    partial = {split: path + ".partial" for split, path in targets.items()}
    for path in partial.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)
    written = {"train": [], "test": []}
    table_rows = {"train": {}, "test": {}}
    try:
        with zipfile.ZipFile(partial["train"], "w") as train_zip, \
                zipfile.ZipFile(partial["test"], "w") as test_zip:
            zips = {"train": train_zip, "test": test_zip}
            for cls, (student, hidden) in plan.items():
                for split, refs in (("train", student), ("test", hidden)):
                    used = set()
                    for ref in refs:                     # one image in memory at a time
                        data, ext = _payload(handle, ref)
                        if handle["root"]:
                            entry = os.path.relpath(ref, handle["root"]).replace(os.sep, "/")
                        else:              # torchvision: <split>/<class>/<file>, or <split>/<file> unlabelled
                            name = (os.path.basename(ref) if isinstance(ref, str)
                                    else f"{ref[0]}_{ref[1]:06d}{ext}")
                            while name in used:
                                name = "_" + name
                            used.add(name)
                            entry = f"{split}/{name}" if src["kind"] == "none" else f"{split}/{cls}/{name}"
                        compress = zipfile.ZIP_STORED if ext in ALREADY_COMPRESSED else zipfile.ZIP_DEFLATED
                        zips[split].writestr(entry, data, compress_type=compress)
                        written[split].append(entry)
                        if src["kind"] == "csv":
                            path, row = src["rows"][ref]
                            table_rows[split].setdefault(path, []).append(row)
            for split, by_table in table_rows.items():
                for path, rows in by_table.items():
                    zips[split].writestr(path, _trimmed_table(src["tables"][path], rows),
                                         compress_type=zipfile.ZIP_DEFLATED)
        for split, path in targets.items():
            os.replace(partial[split], path)
    finally:
        for path in partial.values():
            if os.path.exists(path):
                os.remove(path)

    labelled = src["kind"] != "none"
    layout = {
        "label_source": src["kind"],
        "source_title": src["title"],
        "labelled": labelled,
        "class_order": class_order(plan, numeric=src["kind"] == "csv") if labelled else [],
        "student": _spec(src, written["train"], sorted(table_rows["train"])),
        "hidden": _spec(src, written["test"], sorted(table_rows["test"])),
        "student_archive": images_zip_name(code),
        "hidden_archive": test_images_zip_relname(code),
    }
    return {"official_split": official, "layout": layout}


def _take(handle, refs, wanted, rng, seen) -> list:
    """Up to `wanted` readable images (all if None) in random order, skipping any image whose
    exact content was already used, so no picture can appear twice anywhere."""
    order = list(refs)
    rng.shuffle(order)
    taken = []
    for ref in order:
        if wanted is not None and len(taken) == wanted:
            break
        try:
            data, _ = _payload(handle, ref)
        except Exception:  # noqa: BLE001 - corrupt or unreadable source image
            continue
        digest = hashlib.sha1(data).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        taken.append(ref)
    return taken
