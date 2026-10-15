"""Fetch, inspect and sample real image datasets for CV assignments.

Fixed, tested code instead of model-written code: the model only suggests which
dataset to use; downloading, counting, sampling and zipping happen here, the same
way every run. Images keep their natural size and format: files that exist on disk
are copied byte for byte, and images a dataset only holds in memory (e.g. SVHN,
CIFAR) are saved losslessly as PNG at their original size.

Sources:
    torchvision  a class in torchvision.datasets (all of its splits are loaded)
    kaggle       an "owner/dataset" slug whose images are organised in class folders

Leakage rules (the hidden grading set must never contain what students train on):
- When a dataset ships its own test split, hidden images come from that split and
  student images from the rest, as long as every chosen class has enough of both.
  Otherwise images are pooled and split at random.
- Byte-identical images are used at most once across all classes and both archives.
- A Kaggle download holding several variants of the same photos (e.g. color/,
  grayscale/, segmented/ copies of one class tree) uses only one variant.
"""

from __future__ import annotations

import hashlib
import inspect as _inspect
import io
import os
import random
import re
import tempfile
import zipfile

from PIL import Image

from naming import images_zip_name, test_images_zip_relname

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm", ".pgm", ".tif", ".tiff")
ALREADY_COMPRESSED = (".jpg", ".jpeg", ".png", ".webp")
LABEL_FILE_EXTS = (".csv", ".json", ".mat", ".txt", ".xml")
DOWNLOAD_ROOT = os.path.join(tempfile.gettempdir(), "assignment_pipeline_datasets")
SPLIT_NAMES = {"train": "train", "training": "train", "val": "train", "valid": "train",
               "validation": "train", "trainval": "train", "test": "test", "testing": "test"}
NOT_CLASS_NAMES = {"images", "imgs", "data", "dataset", "raw", "raw-img"} | set(SPLIT_NAMES)

_HANDLES: dict = {}   # (source, source_id) -> loaded dataset, reused between inspect and build


class DatasetUnavailable(Exception):
    """The dataset could not be downloaded or is not laid out in a way we can sample."""


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
    cls = _tv_class(source_id)
    root = os.path.join(DOWNLOAD_ROOT, "torchvision")
    os.makedirs(root, exist_ok=True)
    before = _folder_mb(root)
    parts = _tv_parts(cls, root)
    grown = _folder_mb(root) - before
    items: dict[str, list] = {}
    datasets = [ds for _, ds in parts]
    for p, (tag, ds) in enumerate(parts):
        names = getattr(ds, "classes", None) or getattr(ds, "categories", None)
        paths = _tv_paths(ds)
        for i, label in enumerate(_tv_labels(ds)):
            if label < 0:
                continue          # unlabeled samples (e.g. STL10 'unlabeled')
            name = names[label] if names and label < len(names) else f"class_{label}"
            items.setdefault(_safe_name(name), []).append((paths[i] if paths else (p, i), tag))
    return {"name": cls.__name__, "parts": datasets, "items": items, "label_files": [], "variants": [],
            "download": f"downloaded {grown:.0f} MB" if grown > 1 else "already downloaded"}


# ── kaggle ─────────────────────────────────────────────────────────────────
def _load_kaggle(slug: str) -> dict:
    import kagglehub
    try:
        path = kagglehub.dataset_download(slug)
    except Exception as exc:  # noqa: BLE001 - auth/network/not-found all mean "unavailable"
        raise DatasetUnavailable(f"Kaggle download failed for '{slug}': {exc}") from exc
    root = os.path.normpath(path)
    variants: dict[str, dict] = {}      # variant folder -> class -> [(file, split_tag)]
    label_files, seen = [], set()
    for folder, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__MACOSX")
        rel = os.path.relpath(folder, path)
        if rel.count(os.sep) < 2:
            label_files += [os.path.join(rel, f) for f in files if f.lower().endswith(LABEL_FILE_EXTS)
                            and not f.lower().startswith(("readme", "license"))]
        images = sorted(f for f in files if f.lower().endswith(IMAGE_EXTS) and not f.startswith("."))
        if not images or os.path.normpath(folder) == root:
            continue                  # images loose in the download root have no class folder
        parts = rel.split(os.sep)
        name = _safe_name(parts[-1])
        if name.lower() in NOT_CLASS_NAMES:
            continue                  # e.g. images directly inside train/ with labels elsewhere
        split = next((SPLIT_NAMES[p.lower()] for p in parts[:-1] if p.lower() in SPLIT_NAMES), None)
        variant = "/".join(p for p in parts[:-1] if p.lower() not in SPLIT_NAMES) or "."
        for f in images:
            full = os.path.join(folder, f)
            key = (variant, name, f, os.path.getsize(full))   # duplicate copies of one file
            if key not in seen:
                seen.add(key)
                variants.setdefault(variant, {}).setdefault(name, []).append((full, split))
    if not variants:
        raise DatasetUnavailable(f"'{slug}' has no images organised in class folders. Pick another option.")
    size = {v: sum(len(refs) for refs in classes.values()) for v, classes in variants.items()}
    chosen = max(sorted(variants), key=size.get)       # one variant only, never pooled
    return {"name": slug.split("/")[-1], "parts": None, "items": variants[chosen],
            "label_files": label_files[:5],
            "variants": sorted(variants) if len(variants) > 1 else [],
            "variant_used": chosen,
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
        # A class needs at least one training and one hidden test image.
        handle["items"] = {k: v for k, v in handle["items"].items() if len(v) >= 2}
        if len(handle["items"]) < 2:
            raise DatasetUnavailable(f"'{source_id}' has fewer than two classes with at least 2 images.")
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


def inspect_dataset(source: str, source_id: str) -> dict:
    """Download (or reuse) the dataset and report what is really in it."""
    handle = _handle(source, source_id)
    rng = random.Random(0)
    pool = [ref for entries in handle["items"].values() for ref, _ in entries]
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
        "total": len(pool),
        "classes": [{"name": k, "count": len(v)} for k, v in sorted(handle["items"].items())],
        "size_range": size_range,
        "avg_kb": round(sum(kbs) / len(kbs), 1) if kbs else 0.0,
        "official_test_split": any(tag == "test" for entries in handle["items"].values() for _, tag in entries),
        "label_files": handle.get("label_files", []),
        "variants": handle.get("variants", []),
        "variant_used": handle.get("variant_used", ""),
    }


def default_choice(info: dict, suggested: list) -> dict:
    """Suggested classes that really exist (else the 10 largest), up to 150 images each."""
    counts = {c["name"]: c["count"] for c in info["classes"]}
    wanted = {_safe_name(s).lower() for s in suggested or []}
    classes = [name for name in counts if name.lower() in wanted]
    if len(classes) < 2:
        classes = sorted(counts, key=counts.get, reverse=True)[:10]
    return {"classes": sorted(classes),
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


def _pools(handle, cls) -> tuple[list, list, list]:
    """(official train refs, official test refs, all refs), each in a stable order."""
    entries = sorted(handle["items"][cls], key=lambda e: str(e[0]))
    return ([r for r, tag in entries if tag != "test"], [r for r, tag in entries if tag == "test"],
            [r for r, _ in entries])


def build_archives(source: str, source_id: str, choice: dict, workspace: str, code: str) -> dict:
    """Sample and zip: images_<CODE>.zip (train) and tests/test_images_<CODE>.zip (hidden).
    Images keep their natural size and format; unreadable and duplicate files are skipped.
    Returns {"official_split": bool} describing how the hidden set was chosen."""
    handle = _handle(source, source_id)
    classes = list(choice["classes"])
    per_class = None if choice.get("use_all") else int(choice["images_per_class"])
    fraction = float(choice["test_fraction"])
    wants = {}
    for cls in classes:
        train_refs, test_refs, all_refs = _pools(handle, cls)
        total = len(all_refs) if per_class is None else per_class
        n_test = _test_count(total, fraction)
        wants[cls] = (train_refs, test_refs, all_refs, total - n_test, n_test)
    official = all(test and train and (per_class is None or (len(train) >= n_train and len(test) >= n_test))
                   for train, test, _, n_train, n_test in wants.values())

    rng = random.Random(42)
    seen: set = set()                           # content hashes already used anywhere
    plan = {}
    for cls in classes:                         # choose first, so nothing is written on failure
        train_refs, test_refs, all_refs, n_train, n_test = wants[cls]
        if official:
            hidden = _take(handle, test_refs, None if per_class is None else n_test, rng, seen)
            student = _take(handle, train_refs, None if per_class is None else n_train, rng, seen)
        else:
            picked = _take(handle, all_refs, None if per_class is None else n_train + n_test, rng, seen)
            n_hidden = _test_count(len(picked), fraction) if len(picked) >= 2 else 0
            hidden, student = picked[:n_hidden], picked[n_hidden:]
        if not hidden or not student or (per_class is not None and len(hidden) + len(student) < per_class):
            raise DatasetUnavailable(
                f"Class '{cls}' has only {len(student) + len(hidden)} usable images "
                f"(after skipping unreadable and duplicate files); asked for "
                f"{'every image' if per_class is None else per_class}.")
        plan[cls] = (student, hidden)

    targets = {"train": os.path.join(workspace, images_zip_name(code)),
               "test": os.path.join(workspace, *test_images_zip_relname(code).split("/"))}
    partial = {split: path + ".partial" for split, path in targets.items()}
    for path in partial.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with zipfile.ZipFile(partial["train"], "w") as train_zip, \
                zipfile.ZipFile(partial["test"], "w") as test_zip:
            for cls, (student, hidden) in plan.items():
                for split, refs, zf in (("train", student, train_zip), ("test", hidden, test_zip)):
                    for k, ref in enumerate(refs, 1):          # one image in memory at a time
                        data, ext = _payload(handle, ref)
                        compress = zipfile.ZIP_STORED if ext in ALREADY_COMPRESSED else zipfile.ZIP_DEFLATED
                        zf.writestr(f"{split}/{cls}/img_{k:04d}{ext}", data, compress_type=compress)
        for split, path in targets.items():
            os.replace(partial[split], path)
    finally:
        for path in partial.values():
            if os.path.exists(path):
                os.remove(path)
    return {"official_split": official}


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
