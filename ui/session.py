"""
ui/session.py

Interaction layer that decouples the pipeline's human-in-the-loop points from
the terminal, so the same agent code drives BOTH the CLI and the Streamlit UI.

Agents talk to an `io` object with two methods:
    io.emit(kind, **payload)  -> push progress / an artifact to the user
    io.ask(spec)              -> block until the user answers; returns the answer

Two implementations:
    TerminalIO  -> reproduces the original print()/input() behaviour (CLI).
    WebSession  -> runs the pipeline in a worker thread; ask() blocks on a queue
                   the Streamlit UI fills; emit() records events the UI renders.

Ask `spec` is a dict with a "kind" plus optional fields. Answer conventions
(uppercased for menus, raw for text):
    select_option -> "1".."N" | "M" | "Q"
    recovery      -> "R" | "S" | "B"
    draft_action  -> "A" | "EP" | "ED" | "SD" | "B"
    eval_action   -> "A" | "D" | "R"
    text          -> free text (spec["prompt"])
    discuss       -> free text incl. "accept" / "back"
"""

import queue
import threading
import textwrap


# ── terminal rendering helpers (moved here from problem_dataset) ───────────

def _clip(value, width):
    value = " ".join(str(value or "").split())
    return textwrap.shorten(value, width=width, placeholder="...").ljust(width)


def _render_options(options):
    widths = (3, 34, 25, 34, 9, 10)
    headers = ("#", "Problem Title", "Dataset", "Learning Objective", "Rows", "Balance")
    rule = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print("\n" + rule)
    print("| " + " | ".join(_clip(v, w) for v, w in zip(headers, widths)) + " |")
    print(rule)
    for index, option in enumerate(options, 1):
        row = (index, option.get("title"), option.get("dataset"),
               option.get("learning_objective"), option.get("rows"), option.get("balance"))
        print("| " + " | ".join(_clip(v, w) for v, w in zip(row, widths)) + " |")
    print(rule)
    print("\n[1-{}] Select   [M] More options   [Q] Change topic".format(len(options)))


def _render_draft(plan, statement):
    print("\n" + "=" * 60)
    print("DATASET LOADED AND PREPARED")
    print("=" * 60)
    print(plan or "[Dataset plan was not returned]")
    print("\n" + "-" * 60)
    print("PROBLEM STATEMENT DRAFT:")
    print("-" * 60)
    print(statement or "[Problem statement was not returned]")
    print("-" * 60)


# ── terminal implementation (CLI parity) ───────────────────────────────────

class TerminalIO:
    """Renders to stdout / reads from stdin, matching the original CLI."""

    def emit(self, kind, **payload):
        if kind == "options":
            _render_options(payload.get("options", []))
        elif kind == "draft":
            _render_draft(payload.get("plan", ""), payload.get("statement", ""))
        elif kind in ("log", "research", "test_proposal", "notice"):
            text = payload.get("text", "")
            if text:
                print(text)
        # decorative kinds (stage, files, done) are no-ops in the terminal;
        # the agents already print their own banners.

    def ask(self, spec):
        kind = spec.get("kind")
        if kind == "select_option":
            return input("\nYour choice: ").strip().upper()
        if kind == "recovery":
            return input("Choose: [R] retry same option  [S] synthetic dataset instead  "
                         "[B] back to options: ").strip().upper()
        if kind == "draft_action":
            return input("\nYour choice [A/EP/ED/SD/B]: ").strip().upper()
        if kind == "eval_action":
            return input("\nYour choice [A/D/R]: ").strip().upper()
        prompt = spec.get("prompt", "> ")
        return input(prompt + (" " if not prompt.endswith(" ") else "")).strip()


# ── web implementation (Streamlit-driven, thread + queue bridge) ────────────

class WebSession:
    """
    Runs the pipeline in a background worker thread. The worker calls emit()/ask()
    exactly like the terminal path; ask() blocks the worker until the UI supplies
    an answer via answer(). The Streamlit script only ever reads snapshot() and
    calls answer()/start() from the main thread, never touching st state from the
    worker.
    """

    def __init__(self):
        self.events = []          # ordered emitted events (list of dicts)
        self.pending = None       # current question spec awaiting an answer
        self.status = "idle"      # idle | running | awaiting | done | error
        self.result = None
        self.error = None
        self._answer = queue.Queue(maxsize=1)
        self._lock = threading.RLock()
        self._thread = None

    # -- called by the pipeline (worker thread) --
    def emit(self, kind, **payload):
        with self._lock:
            self.events.append({"kind": kind, **payload})

    def ask(self, spec):
        with self._lock:
            self.pending = dict(spec)
            self.status = "awaiting"
        value = self._answer.get()          # blocks the worker thread
        with self._lock:
            self.pending = None
            self.status = "running"
        return value

    # -- called by the UI (main thread) --
    def answer(self, value):
        # Non-blocking: if an answer is already queued (e.g. a double-click),
        # drop the extra rather than blocking the UI thread.
        try:
            self._answer.put_nowait(value)
        except queue.Full:
            pass

    def start(self, target):
        """target: callable(io) -> result. Runs it in a daemon worker thread."""
        def _run():
            with self._lock:
                self.status = "running"
            try:
                self.result = target(self)
                with self._lock:
                    self.status = "done"
            except Exception as e:  # noqa: BLE001 - surfaced to the UI
                self.error = e
                with self._lock:
                    self.status = "error"

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def snapshot(self):
        with self._lock:
            return {
                "events": list(self.events),
                "pending": dict(self.pending) if self.pending else None,
                "status": self.status,
                "result": self.result,
                "error": repr(self.error) if self.error else None,
            }

    def is_awaiting(self):
        with self._lock:
            return self.status == "awaiting" and self.pending is not None
