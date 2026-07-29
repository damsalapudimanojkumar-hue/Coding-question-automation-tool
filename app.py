"""
app.py — Streamlit UI for the assignment pipeline (Agent 2 flow for now).

Run:
    pip install streamlit
    streamlit run app.py

Covers: Start form -> Research -> Options selection -> Dataset/Problem draft
review (with live markdown preview). Agent 3 (evaluation) + downloads come next.

The pipeline runs in a background worker thread via ui.session.WebSession; this
script only reads snapshot() and calls answer()/start() from the main thread.
"""

import io as _io
import os
import sys
import time
import zipfile

# Keep worker-thread prints (emoji banners) from crashing on Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import streamlit as st

from ui.session import WebSession
from tracing import observe, flush as trace_flush
from claude_client import reset_cost, cost_summary
from agents.wiki_loader import wiki_loader_agent
from agents.research import research_agent
from agents.problem_dataset import problem_dataset_agent
from agents.evaluation_designer import evaluation_designer_agent
from agents.codeeditor_design import codeeditor_design_agent
from agents.codeeditor_generate import codeeditor_generate_agent

st.set_page_config(page_title="DSML Assignment Pipeline", layout="wide")


def build_pipeline(cfg):
    """Return a callable(io) that runs the pipeline for cfg's config_type."""
    @observe(name="assignment_pipeline", as_type="chain")
    def pipeline(io):
        try:
            reset_cost()   # start this run's cost tally from zero
            state = dict(cfg)
            state.update(wiki_loader_agent(state))

            if cfg.get("config_type") == "code_editor_type":
                return _run_codeeditor(state, io, cfg)

            io.emit("log", text=f"Wiki loaded (code {state.get('assignment_code')}). Researching...")
            state.update(research_agent(state))
            io.emit("research", text=state.get("research_output", ""),
                    sites=state.get("research_sites", []))
            io.emit("log", text="Research complete. Generating options...")
            state.update(problem_dataset_agent(state, io))

            if not state.get("approved") or not state.get("problem_statement"):
                io.emit("log", text="Stopped before evaluation (no approved problem statement).")
                return state

            io.emit("log", text="Problem approved. Building evaluation (Agent 3)...")
            state.update(evaluation_designer_agent(state, io))
            return state
        finally:
            trace_flush()   # push any buffered Langfuse traces before the worker ends
    return pipeline


def _run_codeeditor(state, io, cfg):
    """Code-editor flow: Wiki -> (optional Research) -> Design (HITL) -> Generate."""
    io.emit("log", text=f"Wiki loaded (code {state.get('assignment_code')}).")
    if cfg.get("use_research"):
        io.emit("log", text="Researching code-editor sites for inspiration...")
        state.update(research_agent(state))
        io.emit("research", text=state.get("research_output", ""),
                sites=state.get("research_sites", []))
    io.emit("log", text="Designing the question...")
    state.update(codeeditor_design_agent(state, io))

    if not state.get("codeeditor_config"):
        io.emit("log", text="Stopped before generate (no approved config).")
        return state

    io.emit("log", text="Config approved. Generating deliverable...")
    state.update(codeeditor_generate_agent(state, io))
    return state


def zip_workspace(workspace):
    """Zip the assignment folder in memory for download (skips caches)."""
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(workspace):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache")]
            for name in files:
                full = os.path.join(root, name)
                zf.write(full, os.path.relpath(full, workspace))
    buf.seek(0)
    return buf.getvalue()


def latest_event(events, kind):
    found = None
    for ev in events:
        if ev.get("kind") == kind:
            found = ev
    return found


def render_research_body(ev):
    """Render a research brief plus the list of sites Tavily actually explored."""
    st.markdown(ev.get("text", ""))
    sites = ev.get("sites") or []
    if sites:
        st.caption("🔎 Sources explored: " + ", ".join(sites))


def render_cost():
    """Show this run's real OpenRouter cost (works for completed or failed runs)."""
    s = cost_summary()
    if not s["calls"]:
        st.caption("💰 Cost: no model calls recorded this run.")
        return
    if s["cost_usd"] > 0:
        st.markdown(
            f"💰 **Cost:** ${s['cost_usd']:.4f}  ·  {s['calls']} model calls  ·  "
            f"{s['total_tokens']:,} tokens ({s['prompt_tokens']:,} in / {s['completion_tokens']:,} out)"
        )
    else:
        # Calls happened but the provider returned no usage/cost (usage accounting off
        # or stripped by the traced client). Show what we know.
        st.markdown(
            f"💰 **Cost:** {s['calls']} model calls  ·  {s['total_tokens']:,} tokens  "
            f"·  _dollar cost unavailable (no usage data returned)_"
        )


# ── session state ──────────────────────────────────────────────────────────
if "ws" not in st.session_state:
    st.session_state.ws = None

st.title("DSML Assignment Pipeline")
ws = st.session_state.ws

# ── START FORM ───────────────────────────────────────────────────────────
if ws is None:
    st.caption("Start a new assignment. Choose the question format below.")
    # Format selector lives OUTSIDE the form so the fields below adapt to it.
    fmt_label = st.radio(
        "Question format",
        ["Notebook / vscode (dataset + pytest)", "Code-editor (function + test cases)"],
        index=0, horizontal=True,
    )
    is_codeeditor = fmt_label.startswith("Code-editor")

    with st.form("start"):
        if is_codeeditor:
            topic = st.text_input("Topic", "Softmax")
            objective = st.text_area(
                "Learning objective",
                "Implement a numerically stable softmax from scratch.",
            )
            c1, c2 = st.columns(2)
            code = c1.text_input("Assignment code (short, e.g. SFM)", "SFM")
            difficulty = c2.selectbox("Difficulty", ["auto", "EASY", "MEDIUM", "HARD"], index=0)
            c3, c4 = st.columns(2)
            num_tests = c3.slider("Number of test cases", 8, 12, 10)
            num_candidates = c4.slider("Number of questions", 1, 5, 1,
                                       help="More than 1 generates several DIFFERENT questions "
                                            "on the topic for you to pick from.")
            atype = "tabular"   # unused by code-editor; kept for state parity
            st.caption("🔎 Web research (Tavily, scoped to Deep-ML / TensorTonic / "
                       "StrataScratch) runs automatically to find and adapt the best questions.")
        else:
            topic = st.text_input("Topic", "Boosting")
            objective = st.text_area(
                "Learning objective",
                "Apply Boosting to improve predictive performance by sequentially "
                "training weak learners and compare against a single model.",
            )
            col1, col2 = st.columns(2)
            atype = col1.selectbox("Assignment type", ["tabular", "nlp", "cv"], index=0)
            code = col2.text_input("Assignment code (short, e.g. BST)", "BST")
        submitted = st.form_submit_button("Start pipeline")

    if submitted:
        cfg = {
            "topic": topic,
            "learning_objective": objective,
            "assignment_type": atype,
            "config_type": "code_editor_type" if is_codeeditor else "vscode_type",
            "assignment_code": code,
            "use_research": True,   # research always runs (it's the objective)
        }
        if is_codeeditor:
            cfg.update({
                "codeeditor_difficulty": "" if difficulty == "auto" else difficulty,
                "codeeditor_num_tests": num_tests,
                "codeeditor_num_candidates": num_candidates,
                "codeeditor_include_examples": True,   # always on
            })
        st.session_state.is_codeeditor = is_codeeditor
        new_ws = WebSession()
        new_ws.start(build_pipeline(cfg))
        st.session_state.ws = new_ws
        st.rerun()
    st.stop()

# ── RUNNING SESSION ──────────────────────────────────────────────────────
snap = ws.snapshot()
status = snap["status"]
pending = snap["pending"]
events = snap["events"]

with st.sidebar:
    st.caption(f"Status: **{status}**")
    if st.button("Reset / New run"):
        st.session_state.ws = None
        st.session_state.pop("is_codeeditor", None)
        st.rerun()

# research brief (collapsible). For code-editor the brief is shown inside the
# problem-review screen instead, so skip it here to avoid showing it twice.
research = latest_event(events, "research")
if research and research.get("text") and not st.session_state.get("is_codeeditor"):
    with st.expander("Research brief", expanded=False):
        render_research_body(research)

# recent notices
notices = [ev.get("text") for ev in events if ev.get("kind") == "notice"]
for note in notices[-2:]:
    st.info(note)

if status == "error":
    st.error(f"Pipeline error: {snap['error']}")
    render_cost()   # show cost spent before the failure
    st.stop()

if status in ("idle", "running"):
    st.spinner_text = "Working (research / dataset prep can take a minute)..."
    st.info("⏳ Working — research / dataset preparation can take a minute...")
    time.sleep(0.6)
    st.rerun()

if status == "done":
    res = ws.result or {}

    # ── code-editor result view ────────────────────────────────────────────
    if res.get("config_type") == "code_editor_type":
        deliverable = res.get("codeeditor_deliverable")
        if deliverable:
            st.success("Question generated.")
        else:
            st.warning("Stopped before generate (no approved config).")
        render_cost()

        cfg_obj = res.get("codeeditor_config") or {}
        if cfg_obj:
            st.subheader(cfg_obj.get("rephrased_short_text") or cfg_obj.get("short_text", "Question"))
            st.markdown(cfg_obj.get("rephrased_question_text") or cfg_obj.get("question_text", ""))
            with st.expander("Reference solution"):
                st.code(cfg_obj.get("solution_code", ""), language="python")

        outs = latest_event(events, "codeeditor_outputs")
        if outs:
            st.subheader("Computed test outputs (sanity-check)")
            for q in outs.get("questions", []):
                st.caption(f"{q.get('short_text','')} · {q.get('function_name','')}")
                st.table([
                    {"#": c["order"], "hidden": c["is_hidden"],
                     "input": str(c["input"]), "output": str(c["output"])}
                    for c in q.get("cases", [])
                ])

        if deliverable and os.path.isfile(deliverable):
            with open(deliverable, "rb") as fh:
                st.download_button(
                    "⬇️ Download deliverable (.zip)",
                    data=fh.read(),
                    file_name=os.path.basename(deliverable),
                    mime="application/zip",
                )
        readable = res.get("codeeditor_readable_files") or []
        if readable:
            st.caption("Also written next to the zip (readable): " + ", ".join(readable))
        st.caption(f"Saved: {os.path.dirname(deliverable) if deliverable else res.get('output_dir', '')}")
        st.stop()

    if res.get("evaluation_complete"):
        st.success("Pipeline complete — assignment generated.")
    else:
        st.warning("Stopped after Agent 2 (no approved problem statement).")

    render_cost()

    st.subheader("Problem statement (rendered preview)")
    st.markdown(res.get("problem_statement", "[none]"))
    with st.expander("Dataset plan"):
        st.markdown(res.get("dataset_plan", "[none]"))

    files = res.get("generated_files")
    workspace = res.get("output_dir") or res.get("generated_files_path")
    if files:
        st.subheader("Generated files")
        for f in files:
            mark = "✅" if f["exists"] else "❌"
            st.write(f"{mark} `{f['name']}`  ({f['size_kb']} KB)")
        if res.get("question_file"):
            st.caption(f"Question file: {res['question_file']}")
        if workspace and os.path.isdir(workspace):
            st.download_button(
                "⬇️ Download assignment bundle (.zip)",
                data=zip_workspace(workspace),
                file_name=f"{os.path.basename(workspace)}_bundle.zip",
                mime="application/zip",
            )
    st.caption(f"Files saved in: {workspace or ''}")
    st.stop()

# ── status == awaiting: render the widget for the pending question ─────────
kind = pending["kind"]

if kind == "select_option":
    st.subheader("Choose a problem / dataset option")
    opts_ev = latest_event(events, "options")
    options = opts_ev["options"] if opts_ev else []
    for i, o in enumerate(options, 1):
        with st.container(border=True):
            st.markdown(
                f"**{i}. {o.get('title')}**  \n"
                f"Dataset: {o.get('dataset')}  ·  Rows: {o.get('rows')}  ·  Balance: {o.get('balance')}"
            )
            st.caption(o.get("learning_objective", ""))
            if st.button(f"Select #{i}", key=f"sel_{i}_{len(options)}"):
                ws.answer(str(i))
                st.rerun()
    c1, c2 = st.columns(2)
    if c1.button("More options", key="more"):
        ws.answer("M")
        st.rerun()
    if c2.button("Change topic (stop)", key="quit"):
        ws.answer("Q")
        st.rerun()

elif kind == "recovery":
    st.warning("Dataset preparation did not complete for this option.")
    c1, c2, c3 = st.columns(3)
    if c1.button("Retry same option", key="rec_r"):
        ws.answer("R")
        st.rerun()
    if c2.button("Synthetic dataset", key="rec_s"):
        ws.answer("S")
        st.rerun()
    if c3.button("Back to options", key="rec_b"):
        ws.answer("B")
        st.rerun()

elif kind == "draft_action":
    draft = latest_event(events, "draft") or {}
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Dataset plan")
        st.markdown(draft.get("plan", "") or "[no plan]")
    with right:
        st.subheader("Problem statement (preview)")
        st.markdown(draft.get("statement", "") or "[no statement]")
    st.divider()
    c = st.columns(5)
    if c[0].button("Approve", key="d_a", type="primary"):
        ws.answer("A")
        st.rerun()
    if c[1].button("Edit problem", key="d_ep"):
        ws.answer("EP")
        st.rerun()
    if c[2].button("Edit dataset", key="d_ed"):
        ws.answer("ED")
        st.rerun()
    if c[3].button("Synthetic", key="d_sd"):
        ws.answer("SD")
        st.rerun()
    if c[4].button("Back to options", key="d_b"):
        ws.answer("B")
        st.rerun()

elif kind == "edit_dataset":
    st.subheader("Edit dataset")
    st.caption("Presets run as deterministic code. Ground truth is never changed; "
               "size/noise/nulls apply to train only; dropped columns are removed from train + test.")
    atype = pending.get("assignment_type", "tabular")
    columns = pending.get("columns", [])
    with st.form("edit_form"):
        resize = st.slider(
            "Train size factor  (1.0 = unchanged, <1 subsample, >1 bootstrap-duplicate)",
            0.25, 2.0, 1.0, 0.05,
        )
        rebalance = st.checkbox("Rebalance classes (down-sample to the smallest class)")
        add_noise, inject_nulls, drop_cols, impute = 0.0, 0.0, [], "none"
        if atype == "tabular":
            impute = st.selectbox(
                "Impute missing values in train (default: leave nulls for the student)",
                ["none", "median", "mean", "most_frequent"], index=0,
            )
            add_noise = st.slider("Add Gaussian noise to numeric features (intensity × std)",
                                  0.0, 0.5, 0.0, 0.05)
            inject_nulls = st.slider("Inject nulls into train features (fraction of cells)",
                                     0.0, 0.3, 0.0, 0.05)
            drop_cols = st.multiselect("Drop feature columns (removed from train + test)", columns)
        freeform = st.text_area("Extra instructions (optional, freeform → model edit)", "")
        c1, c2 = st.columns(2)
        apply_clicked = c1.form_submit_button("Apply", type="primary")
        cancel_clicked = c2.form_submit_button("Cancel / back")

    if cancel_clicked:
        ws.answer({"cancel": True})
        st.rerun()
    if apply_clicked:
        ws.answer({
            "cancel": False,
            "resize_factor": resize,
            "rebalance": rebalance,
            "add_noise": add_noise,
            "inject_nulls": inject_nulls,
            "drop_columns": drop_cols,
            "impute": None if impute == "none" else impute,
            "freeform": freeform,
        })
        st.rerun()

elif kind == "eval_action":
    st.subheader("Review proposed test cases")
    proposal = latest_event(events, "test_proposal")
    if proposal and proposal.get("text"):
        st.code(proposal["text"])
    c1, c2, c3 = st.columns(3)
    if c1.button("Approve → generate files", key="ev_a", type="primary"):
        ws.answer("A")
        st.rerun()
    if c2.button("Discuss / refine", key="ev_d"):
        ws.answer("D")
        st.rerun()
    if c3.button("Regenerate", key="ev_r"):
        ws.answer("R")
        st.rerun()

elif kind == "codeeditor_pick":
    cand_ev = latest_event(events, "codeeditor_candidates") or {}
    candidates = cand_ev.get("candidates", [])
    rbrief = latest_event(events, "research")
    if rbrief and rbrief.get("text"):
        with st.expander("🔎 Research (Tavily) — read before choosing", expanded=False):
            render_research_body(rbrief)
    st.subheader(f"Pick a problem ({len(candidates)} candidates)")
    for c in candidates:
        with st.container(border=True):
            st.markdown(f"**{c.get('index')}. {c.get('short_text','')}**  "
                        f"· `{c.get('function_name','')}` · {c.get('difficulty','') or 'auto'}")
            if c.get("focus"):
                st.caption("What's different: " + c["focus"])
            qt = c.get("question_text", "")
            st.markdown(qt[:700] + ("…" if len(qt) > 700 else ""))
            if c.get("problems"):
                st.caption("⚠ " + "; ".join(c["problems"]))
            if st.button(f"Choose #{c.get('index')}", key=f"pick_{c.get('index')}", type="primary"):
                ws.answer(str(c.get("index")))
                st.rerun()
    if st.button("Regenerate all", key="pick_regen"):
        ws.answer("R")
        st.rerun()
    with st.form("pick_revise", clear_on_submit=True):
        notes = st.text_area("Or describe what you want and regenerate:", "")
        if st.form_submit_button("Regenerate with notes") and notes.strip():
            ws.answer(notes)
            st.rerun()

elif kind == "codeeditor_review":
    prev = latest_event(events, "codeeditor_preview") or {}
    cfg_obj = prev.get("config", {})
    problems = prev.get("problems", [])
    phase = pending.get("phase") or prev.get("phase") or "tests"

    if phase == "problem":
        # show the Tavily research brief right here, before the problem statement
        rbrief = latest_event(events, "research")
        if rbrief and rbrief.get("text"):
            with st.expander("🔎 Research (Tavily) — read before reviewing the problem",
                             expanded=True):
                render_research_body(rbrief)

        st.subheader("Review 1 of 2 — Problem statement")
        if cfg_obj:
            st.markdown(f"**{cfg_obj.get('rephrased_short_text') or cfg_obj.get('short_text','')}**")
            st.caption(
                f"{cfg_obj.get('function_name','')}"
                f"({', '.join(cfg_obj.get('param_names', []))})  ·  "
                f"{cfg_obj.get('library','numpy')}  ·  {cfg_obj.get('difficulty','EASY')}"
            )
            st.markdown(cfg_obj.get("rephrased_question_text") or cfg_obj.get("question_text", ""))
            with st.expander("Reference solution", expanded=True):
                st.code(cfg_obj.get("solution_code", ""), language="python")
            with st.expander("Starter code"):
                st.code(cfg_obj.get("starter_code", ""), language="python")
        else:
            st.info("The model's problem response could not be parsed — regenerate or send notes.")
    else:  # tests
        st.subheader("Review 2 of 2 — Test cases")
        st.caption(
            f"{cfg_obj.get('short_text','')}  ·  {cfg_obj.get('function_name','')}"
            f"({', '.join(cfg_obj.get('param_names', []))})"
        )
        cases = prev.get("cases") or []
        if cases:
            st.markdown("**Test inputs → computed outputs** (from running the reference solution):")
            st.table([
                {"#": c["order"], "hidden": c["is_hidden"], "weight": c["weightage"],
                 "input": str(c["input"]), "output": str(c["output"])}
                for c in cases
            ])
        else:
            tds = cfg_obj.get("test_definitions") or []
            st.markdown(f"**Test cases ({len(tds)})** — inputs only:")
            st.table([
                {"#": i, "hidden": t.get("is_hidden", False),
                 "weight": t.get("weightage", 10), "inputs": str(t.get("inputs"))}
                for i, t in enumerate(tds, 1)
            ])

    if problems:
        st.warning("Warnings:\n" + "\n".join(f"- {p}" for p in problems))

    approve_label = "Approve → design test cases" if phase == "problem" else "Approve → generate"
    c1, c2 = st.columns(2)
    if c1.button(approve_label, key="ce_a", type="primary"):
        ws.answer("A")
        st.rerun()
    if c2.button("Regenerate fresh", key="ce_r"):
        ws.answer("R")
        st.rerun()
    with st.form("ce_revise", clear_on_submit=True):
        notes = st.text_area("Or describe a revision (submit to re-design):", "")
        if st.form_submit_button("Submit revision") and notes.strip():
            ws.answer(notes)
            st.rerun()

elif kind == "text":
    st.subheader(pending.get("prompt", "Enter details"))
    proposal = latest_event(events, "test_proposal")
    if proposal and proposal.get("text"):
        with st.expander("Current test-case proposal", expanded=False):
            st.code(proposal["text"])
    with st.form("text_form", clear_on_submit=True):
        txt = st.text_area("Your input", "")
        if st.form_submit_button("Submit"):
            ws.answer(txt)
            st.rerun()

else:
    st.write("Waiting for:", pending)
