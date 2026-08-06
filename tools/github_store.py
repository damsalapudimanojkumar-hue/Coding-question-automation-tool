"""
tools/github_store.py

A durable question LIBRARY backed by a GitHub branch.

Why this file exists
--------------------
This is the SINGLE place that talks to the GitHub API - the same way claude_client.py
is the single door to the LLM. The rest of the app calls a few plain functions here
(push_question / list_questions / fetch_folder) and never touches HTTP itself. If we
ever move the library to S3, only THIS file changes.

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
            boosting_1_20260803-152230/  solution.ipynb, tests/..., question.json, *.csv

- Files are stored LOOSE (readable in GitHub), not as a zip. The app zips on download.
- Folder name = <topic>_<N>_<timestamp>: the number is human-readable ("2nd softmax") and
  the timestamp guarantees no two folders ever collide, even if the peek races.

The GitHub mechanic (Contents API), in three moves
--------------------------------------------------
- CREATE a file  = PUT /repos/{owner}/{repo}/contents/{path}   body {message, content(base64), branch}
                   -> makes the file AND a commit in one call.
- LIST a folder  = GET /repos/{owner}/{repo}/contents/{dir}?ref={branch}
- READ a file    = GET its download_url (raw bytes).
Every call carries the token in an Authorization header, and every WRITE names
branch=GH_BRANCH so it can never touch the deployed branch (no redeploy loop).

Configuration (read the same way as OPENROUTER_API_KEY - os.getenv + .env / Streamlit secrets)
---------------------------------------------------------------------------------------------
    GITHUB_TOKEN   fine-grained token, scoped to this repo, Contents: Read and write
    GH_REPO        "owner/repo"  (e.g. "damsalapudimanojkumar-hue/assignment_pipeline")
    GH_BRANCH      branch to store into (default "question-bank"). Must already exist.
"""

import os
import re
import base64
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

_API = "https://api.github.com"
_TIMEOUT = 30


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


def _put_file(path: str, content_bytes: bytes, message: str):
    """Create (or overwrite) one file at `path` on GH_BRANCH via the Contents API."""
    token, repo, branch = _cfg()
    b64 = base64.b64encode(content_bytes).decode("ascii")
    body = {"message": message, "content": b64, "branch": branch}
    r = requests.put(f"{_API}/repos/{repo}/contents/{path}",
                     headers=_headers(), json=body, timeout=_TIMEOUT)
    if r.status_code == 422:
        # Most common cause: the branch does not exist yet.
        raise RuntimeError(
            f"GitHub rejected the write (422). Does the '{branch}' branch exist? "
            f"Create it once, then retry. Detail: {r.text[:200]}"
        )
    r.raise_for_status()
    return r.json()


def _api(method: str, endpoint: str, **kwargs):
    """Call GitHub's Git Data API for one atomic folder-level change."""
    _, repo, _ = _cfg()
    response = requests.request(method, f"{_API}/repos/{repo}{endpoint}",
                                headers=_headers(), timeout=_TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json()


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
    """Push one question to <kind>/<topic>_<N>_<ts>/. `files` maps a relative path
    (e.g. 'coding_questions.json' or 'tests/conftest.py') to str or bytes content.
    Returns the folder path created. Raises if not configured."""
    token, repo, branch = _cfg()
    if not (token and repo):
        raise RuntimeError(
            "GITHUB_TOKEN / GH_REPO are not set. Add them to .env (local) or "
            "Streamlit secrets (deployed) - see .env.example."
        )
    if not files:
        raise ValueError("No files to push.")
    folder = next_folder_name(kind, topic)
    base = f"{kind}/{folder}"
    for relpath, content in files.items():
        if isinstance(content, str):
            content = content.encode("utf-8")
        _put_file(f"{base}/{relpath}", content, message=f"Add question {base}/{relpath}")
    return base


def list_questions(kind: str) -> list:
    """List the question folders under a kind ('code_editor' | 'vscode'), newest last
    is not guaranteed by the API, so callers may sort by name. Returns [{name, path}]."""
    return [{"name": d["name"], "path": d["path"]}
            for d in _list_dir(kind) if d.get("type") == "dir"]


def delete_question(kind: str, folder_path: str) -> None:
    """Atomically remove one saved question folder from the configured library branch.

    The GitHub Contents API deletes files one at a time, which could leave a
    half-deleted vscode workspace if a request failed. This instead creates one
    commit that removes every file below the folder together.
    """
    token, repo, branch = _cfg()
    if not (token and repo):
        raise RuntimeError("GITHUB_TOKEN / GH_REPO are not set.")
    if kind not in {"code_editor", "vscode"}:
        raise ValueError("Unknown question kind.")

    expected_prefix = f"{kind}/"
    if not folder_path.startswith(expected_prefix) or ".." in folder_path.split("/"):
        raise ValueError("Invalid saved-question folder path.")

    ref = _api("GET", f"/git/ref/heads/{branch}")
    commit_sha = ref["object"]["sha"]
    commit = _api("GET", f"/git/commits/{commit_sha}")
    tree = _api("GET", f"/git/trees/{commit['tree']['sha']}",
                params={"recursive": "1"})
    prefix = f"{folder_path.rstrip('/')}/"
    matches = [entry for entry in tree.get("tree", [])
               if entry.get("type") == "blob" and entry.get("path", "").startswith(prefix)]
    if not matches:
        raise ValueError("This saved question no longer exists in the library.")

    new_tree = _api("POST", "/git/trees", json={
        "base_tree": commit["tree"]["sha"],
        "tree": [{"path": entry["path"], "mode": entry["mode"],
                  "type": "blob", "sha": None} for entry in matches],
    })
    new_commit = _api("POST", "/git/commits", json={
        "message": f"Delete saved question {folder_path}",
        "tree": new_tree["sha"],
        "parents": [commit_sha],
    })
    _api("PATCH", f"/git/refs/heads/{branch}", json={"sha": new_commit["sha"], "force": False})


def fetch_folder(path: str) -> dict:
    """Return {relative_path: bytes} for every file under a question folder, recursing
    into subfolders (so nested vscode workspaces come back whole)."""
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
