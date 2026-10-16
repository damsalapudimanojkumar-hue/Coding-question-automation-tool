"""How an image dataset is laid out inside one archive, described once and used everywhere.

Datasets keep the layout they come with; nothing is renamed or restructured. A layout spec
says where the images are and where their labels come from:

    {"kind": "folders", "roots": ["flowers"]}
        every image sits in <root>/<class name>/<file>; the folder name is the label
    {"kind": "csv", "tables": [{"path": "dataset/train.csv", "columns": [...],
                                "file_column": "image_ID", "label_column": "label",
                                "base": "dataset/train", "suffix": ""}]}
        the dataset's own label table: each row names one image (file_column value + suffix,
        relative to base; base "*" means the name is looked up anywhere in the archive)
        and gives its label
    {"kind": "none", "dirs": ["dataset/train"]}
        images only, no labels

iter_dataset() is the single reader for all of them. The pipeline's own checker runs it,
and its source code is handed to the evaluation agent, so grading reads the hidden images
exactly the way the checker verified them.
"""

from __future__ import annotations

import csv
import inspect
import json
import os

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm", ".pgm", ".tif", ".tiff", ".gif")


def iter_dataset(root, spec):
    """Yield (image path, label) for every image `spec` describes inside the extracted
    archive at `root`. label is a string, or None for an unlabelled dataset."""
    def join(rel):
        return os.path.join(root, *[p for p in rel.split("/") if p])

    if spec["kind"] == "folders":
        for top in spec["roots"]:
            for label in sorted(os.listdir(join(top))):
                folder = os.path.join(join(top), label)
                if os.path.isdir(folder):
                    for name in sorted(os.listdir(folder)):
                        if name.lower().endswith(IMAGE_EXTS):
                            yield os.path.join(folder, name), label
    elif spec["kind"] == "csv":
        by_name = None
        for table in spec["tables"]:
            if table["base"] == "*" and by_name is None:
                by_name = {f: os.path.join(d, f) for d, _, files in os.walk(root) for f in files}
            with open(join(table["path"]), newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    name = row[table["file_column"]].strip() + table["suffix"]
                    if table["base"] == "*":
                        path = by_name.get(name.replace("\\", "/").split("/")[-1], join(name))
                    else:
                        path = join(table["base"] + "/" + name.replace("\\", "/"))
                    yield path, row[table["label_column"]].strip()
    else:
        for top in spec["dirs"]:
            for name in sorted(os.listdir(join(top))):
                if name.lower().endswith(IMAGE_EXTS):
                    yield os.path.join(join(top), name), None


def class_order(labels, numeric: bool = True) -> list:
    """The order class indices follow. Folder names sort as text, exactly like torchvision's
    ImageFolder; table labels that are all whole numbers sort as numbers (0, 1, 2, ..., 10)."""
    labels = sorted(set(labels))
    if numeric and labels and all(str(x).lstrip("-").isdigit() for x in labels):
        return sorted(labels, key=int)
    return labels


def _quote_list(items, limit=12):
    shown = ", ".join(f"`{x}`" for x in items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")


def describe_archive(spec: dict, archive: str) -> str:
    """Plain description of what one archive contains and where its labels are."""
    if spec["kind"] == "folders":
        roots = [r or "(the archive root)" for r in spec["roots"]]
        where = (f"the folder `{roots[0]}/`" if len(roots) == 1
                 else "the folders " + _quote_list([r + "/" for r in roots]))
        return (f"`{archive}` contains {where} with one subfolder per class; every image inside "
                f"a class subfolder belongs to that class (the folder name is the label).")
    if spec["kind"] == "csv":
        lines = []
        for t in spec["tables"]:
            if t["base"] == "*":
                located = "a file name; the image with that name is somewhere in the archive"
            elif t.get("names_only"):
                located = f"the file name of an image inside `{t['base'] or '(the archive root)'}/`"
            else:
                located = f"a path relative to `{t['base'] or '(the archive root)'}/`"
            if t["suffix"]:
                located += f", written without its `{t['suffix']}` extension"
            lines.append(f"`{t['path']}` (columns: {', '.join(t['columns'])}): each row is one image; "
                         f"`{t['file_column']}` gives {located}, and `{t['label_column']}` gives its class.")
        return f"`{archive}` contains the images plus the dataset's own label table(s):\n" + \
               "\n".join(f"  - {line}" for line in lines)
    return (f"`{archive}` contains images only, with no labels, in "
            + _quote_list([d + "/" if d else "(the archive root)" for d in spec["dirs"]]) + ".")


def describe_layout(layout: dict) -> str:
    """Verified layout facts for the problem statement and the evaluation agent."""
    lines = [describe_archive(layout["student"], layout["student_archive"]),
             describe_archive(layout["hidden"], layout["hidden_archive"]) + " (hidden, graders only)"]
    if layout["labelled"]:
        same = (" (the same order torchvision.datasets.ImageFolder uses)"
                if layout["label_source"] == "folders" else "")
        lines.append(f"Class index order (index i is the i-th name){same}: {json.dumps(layout['class_order'])}")
    return "\n".join(lines)


def loader_source() -> str:
    """Source code of the shared reader, for pasting into conftest.py."""
    return "import csv\nimport os\n\nIMAGE_EXTS = " + repr(IMAGE_EXTS) + "\n\n\n" + inspect.getsource(iter_dataset)
