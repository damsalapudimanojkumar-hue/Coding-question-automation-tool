"""
tools/github_store.py

A durable question LIBRARY backed by a GitHub branch.

Why this file exists
--------------------
This is the SINGLE place that talks to the GitHub API - the same way claude_client.py
is the single door to the LLM. The rest of the app calls a few plain functions here
(push_question / list_questions / fetch_folder / resolve_assets / delete_question) and
never touches HTTP itself. If we ever move the library to S3, only THIS file changes.

The problem it solves
---------------------
On Streamlit Cloud the app's disk is ephemeral: the outputs/ folder is wiped on every
redeploy/restart, so questions saved there are not durable. GitHub is durable and free,
so once a question is created we push its files to a dedicated branch and they live there
forever, versioned.

How it stores things
--------------------
    <GH_BRANCH>/
        code_editor/
            softmax_1_20260803-141205/   coding_questions.json, question_sets_questions.json
        vscode/
            boosting_1_20260803-152230/  solution.ipynb, tests/..., <id>.json, *.csv,
                                         images_CNN.zip.asset.json  (pointer, see below)

- Files are stored LOOSE (readable in GitHub), not as a zip. The app zips on download.
- Folder name = <topic>_<N>_<timestamp>: the number is human-readable ("2nd softmax") and
  the timestamp guarantees no two folders ever collide, even if the peek races.
- Files over 1 MB (e.g. image datasets) are not committed to the branch: GitHub's file APIs
  reject large files and every committed byte stays in git history forever. They are
  uploaded as assets of one release (tag "question-bank-assets", up to 2 GB per file), and
  the folder gets a small "<file>.asset.json" pointer {asset_id, name, size, sha256}.
- A save is all-or-nothing: assets are uploaded first, then every small file and pointer
  lands in ONE commit (Git Data API). If anything fails, uploaded assets are removed, so a
  half-saved question can never appear in the library.

Configuration (read the same way as OPENROUTER_API_KEY - os.getenv + .env / Streamlit secrets)
---------------------------------------------------------------------------------------------
    GITHUB_TOKEN   fine-grained token, scoped to this repo, Contents: Read and write
                   (Contents also covers releases and their assets)
    GH_REPO        "owner/repo"  (e.g. "damsalapudimanojkumar-hue/assignment_pipeline")
    GH_BRANCH      branch to store into (default "question-bank"). Must already exist.
"""

import base64
import hashlib
import json
import os
import re
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

_API = "https://api.github.com"
_TIMEOUT = 30
_TRANSFER_TIMEOUT = 600                 # uploads/downloads of large assets
ASSET_TAG = "question-bank-assets"
ASSET_MIN_BYTES = 1024 * 1024           # files above 1 MB are stored as release assets
POINTER_SUFFIX = ".asset.json"


# ── configuration ──────────────────────────────────────────────────────────
def _cfg():
    # re-read .env each call so a token added after the app started is picked up
    # (same trick web_research.py uses for the Tavily key).
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GH_REPO")                 # "owner/repo"
    branch = os.getenv("GH_BRANCH", "question-bank")
    return token, repo, branch


def available() -> bool:
    """True only when a token + repo are configured. The UI uses this to decide whether
    to show the 'Save to Library' button at all."""
    token, repo, _ = _cfg()
    return bool(token and repo)


def _headers():
    token, _, _ = _cfg()
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _slug(text: str) -> str:
    """A safe folder-name slug: lowercase, non-alphanumerics -> underscore."""
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_") or "question"


# ── low-level API helpers ──────────────────────────────────────────────────
def _reason(response) -> str:
    """GitHub's own explanation of a failed request."""
    try:
        message = response.json().get("message", "")
    except ValueError:
        message = response.text[:200]
    return f"GitHub {response.status_code}: {message}"


def _api(method: str, endpoint: str, **kwargs):
    """One call to the repository REST API; raises with GitHub's own reason on failure."""
    _, repo, _ = _cfg()
    response = requests.request(method, f"{_API}/repos/{repo}{endpoint}",
                                headers=_headers(), timeout=kwargs.pop("timeout", _TIMEOUT), **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"{method} {endpoint} failed. {_reason(response)}")
    return response.json() if response.content else {}


def _list_dir(path: str) -> list:
    """GET one directory's contents. Returns a list of {name, path, type, download_url}
    dicts, or [] if the path doesn't exist yet (404)."""
    token, repo, branch = _cfg()
    r = requests.get(f"{_API}/repos/{repo}/contents/{path}",
                     headers=_headers(), params={"ref": branch}, timeout=_TIMEOUT)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def _branch_head() -> tuple[str, dict]:
    """(commit sha, commit) at the tip of the library branch."""
    _, repo, branch = _cfg()
    r = requests.get(f"{_API}/repos/{repo}/git/ref/heads/{branch}", headers=_headers(), timeout=_TIMEOUT)
    if r.status_code == 404:
        raise RuntimeError(f"The '{branch}' branch does not exist in {repo}. Create it once, then retry.")
    if r.status_code >= 400:
        raise RuntimeError(f"Could not read the '{branch}' branch. {_reason(r)}")
    sha = r.json()["object"]["sha"]
    return sha, _api("GET", f"/git/commits/{sha}")


def _commit(tree_entries: list, message: str) -> None:
    """Apply tree changes to the library branch as ONE commit (all-or-nothing)."""
    _, _, branch = _cfg()
    head_sha, head = _branch_head()
    tree = _api("POST", "/git/trees", json={"base_tree": head["tree"]["sha"], "tree": tree_entries})
    commit = _api("POST", "/git/commits", json={"message": message, "tree": tree["sha"],
                                                "parents": [head_sha]})
    _api("PATCH", f"/git/refs/heads/{branch}", json={"sha": commit["sha"], "force": False})


# ── large files: release assets ────────────────────────────────────────────
def _assets_release() -> dict:
    """The release whose assets hold large library files; created on first use."""
    _, repo, branch = _cfg()
    r = requests.get(f"{_API}/repos/{repo}/releases/tags/{ASSET_TAG}", headers=_headers(), timeout=_TIMEOUT)
    if r.status_code == 200:
        return r.json()
    if r.status_code != 404:
        raise RuntimeError(f"Could not read the library's asset storage. {_reason(r)}")
    return _api("POST", "/releases", json={
        "tag_name": ASSET_TAG, "target_commitish": branch, "name": "Question bank assets",
        "body": "Large files (such as image datasets) of questions saved to the Question Library. "
                "Managed by the assignment pipeline app; do not edit or delete by hand.",
        "prerelease": True,
    })


def _upload_asset(release: dict, name: str, data: bytes) -> dict:
    upload_url = release["upload_url"].split("{")[0]
    r = requests.post(upload_url, params={"name": name}, data=data, timeout=_TRANSFER_TIMEOUT,
                      headers={**_headers(), "Content-Type": "application/octet-stream"})
    if r.status_code >= 400:
        raise RuntimeError(f"Uploading {name} ({len(data) / 1048576:.1f} MB) failed. {_reason(r)}")
    return r.json()


def _delete_asset(asset_id: int) -> None:
    _api("DELETE", f"/releases/assets/{asset_id}")


def _asset_name(folder: str, relpath: str) -> str:
    """Unique, URL-safe asset name, e.g. vscode__cnn_1_20261015-165725__images_CV5.zip"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{folder}/{relpath}".replace("/", "__"))


def _download_asset(pointer: dict) -> bytes:
    _, repo, _ = _cfg()
    r = requests.get(f"{_API}/repos/{repo}/releases/assets/{pointer['asset_id']}",
                     headers={**_headers(), "Accept": "application/octet-stream"},
                     timeout=_TRANSFER_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"Downloading {pointer.get('name')} failed. {_reason(r)}")
    if hashlib.sha256(r.content).hexdigest() != pointer.get("sha256"):
        raise RuntimeError(f"{pointer.get('name')} did not match its saved checksum.")
    return r.content


# ── public API used by the app ─────────────────────────────────────────────
def next_folder_name(kind: str, topic: str) -> str:
    """Build '<topic>_<N>_<timestamp>'. N = 1 + the highest existing N for this topic
    inside this kind folder (the 'peek'); the timestamp makes it collision-proof."""
    slug = _slug(topic)
    pat = re.compile(rf"^{re.escape(slug)}_(\d+)_")
    highest = 0
    for item in _list_dir(kind):
        if item.get("type") == "dir":
            m = pat.match(item["name"])
            if m:
                highest = max(highest, int(m.group(1)))
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{slug}_{highest + 1}_{ts}"


def push_question(kind: str, topic: str, files: dict) -> str:
    """Save one question to <kind>/<topic>_<N>_<ts>/ in a single commit. `files` maps a
    relative path (e.g. 'coding_questions.json' or 'tests/conftest.py') to str or bytes.
    Files over ASSET_MIN_BYTES become release assets with a pointer file in the folder.
    Returns the folder path created. Raises (leaving nothing behind) on any failure."""
    token, repo, branch = _cfg()
    if not (token and repo):
        raise RuntimeError(
            "GITHUB_TOKEN / GH_REPO are not set. Add them to .env (local) or "
            "Streamlit secrets (deployed) - see .env.example."
        )
    if not files:
        raise ValueError("No files to push.")
    _branch_head()                          # fail fast on a missing branch, before any upload
    folder = next_folder_name(kind, topic)
    base = f"{kind}/{folder}"
    data = {rel: (c.encode("utf-8") if isinstance(c, str) else c) for rel, c in files.items()}
    large = {rel: content for rel, content in data.items() if len(content) > ASSET_MIN_BYTES}
    uploaded = []
    try:
        committed = {rel: content for rel, content in data.items() if rel not in large}
        if large:
            release = _assets_release()
            for rel, content in large.items():
                asset = _upload_asset(release, _asset_name(base, rel), content)
                uploaded.append(asset["id"])
                committed[rel + POINTER_SUFFIX] = json.dumps({
                    "asset_id": asset["id"], "name": asset["name"], "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }, indent=2).encode("utf-8")
        entries = []
        for rel, content in committed.items():
            blob = _api("POST", "/git/blobs", json={"content": base64.b64encode(content).decode("ascii"),
                                                    "encoding": "base64"})
            entries.append({"path": f"{base}/{rel}", "mode": "100644", "type": "blob", "sha": blob["sha"]})
        _commit(entries, f"Add question {base}")
    except Exception:
        for asset_id in uploaded:             # never leave orphan assets behind
            try:
                _delete_asset(asset_id)
            except Exception:  # noqa: BLE001 - best effort; the original error matters more
                pass
        raise
    return base


def list_questions(kind: str) -> list:
    """List the question folders under a kind ('code_editor' | 'vscode'), newest last
    is not guaranteed by the API, so callers may sort by name. Returns [{name, path}]."""
    return [{"name": d["name"], "path": d["path"]}
            for d in _list_dir(kind) if d.get("type") == "dir"]


def delete_question(kind: str, folder_path: str) -> None:
    """Remove one saved question folder in ONE commit, then delete its release assets."""
    token, repo, branch = _cfg()
    if not (token and repo):
        raise RuntimeError("GITHUB_TOKEN / GH_REPO are not set.")
    if kind not in {"code_editor", "vscode"}:
        raise ValueError("Unknown question kind.")

    expected_prefix = f"{kind}/"
    if not folder_path.startswith(expected_prefix) or ".." in folder_path.split("/"):
        raise ValueError("Invalid saved-question folder path.")

    _, head = _branch_head()
    tree = _api("GET", f"/git/trees/{head['tree']['sha']}", params={"recursive": "1"})
    prefix = f"{folder_path.rstrip('/')}/"
    matches = [entry for entry in tree.get("tree", [])
               if entry.get("type") == "blob" and entry.get("path", "").startswith(prefix)]
    if not matches:
        raise ValueError("This saved question no longer exists in the library.")

    asset_ids = []
    for entry in matches:
        if entry["path"].endswith(POINTER_SUFFIX):
            blob = _api("GET", f"/git/blobs/{entry['sha']}")
            asset_ids.append(json.loads(base64.b64decode(blob["content"]))["asset_id"])

    _commit([{"path": entry["path"], "mode": entry["mode"], "type": "blob", "sha": None}
             for entry in matches], f"Delete saved question {folder_path}")
    for asset_id in asset_ids:
        try:
            _delete_asset(asset_id)
        except Exception:  # noqa: BLE001 - the question is gone; a leftover asset is harmless
            pass


def fetch_folder(path: str) -> dict:
    """Return {relative_path: bytes} for every file under a question folder, recursing
    into subfolders (so nested vscode workspaces come back whole). Large files come back
    as their small "<file>.asset.json" pointers; call resolve_assets() before zipping."""
    out = {}

    def _walk(dir_path, prefix):
        for item in _list_dir(dir_path):
            if item.get("type") == "dir":
                _walk(item["path"], f"{prefix}{item['name']}/")
            elif item.get("type") == "file":
                r = requests.get(item["download_url"], timeout=_TIMEOUT)
                r.raise_for_status()
                out[f"{prefix}{item['name']}"] = r.content

    _walk(path, "")
    return out


def has_assets(files: dict) -> bool:
    return any(rel.endswith(POINTER_SUFFIX) for rel in files)


def resolve_assets(files: dict) -> dict:
    """Replace every "<file>.asset.json" pointer with the real file, checksum-verified."""
    out = {}
    for rel, content in files.items():
        if rel.endswith(POINTER_SUFFIX):
            out[rel[: -len(POINTER_SUFFIX)]] = _download_asset(json.loads(content))
        else:
            out[rel] = content
    return out
