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
from agents.wiki_loader import wiki_loader_agent
from agents.research import research_agent
from agents.problem_dataset import problem_dataset_agent
from agents.evaluation_designer import evaluation_designer_agent

st.set_page_config(page_title="DSML Assignment Pipeline", layout="wide")


def build_pipeline(cfg):
    """Return a callable(io) that runs Wiki -> Research -> Agent 2 -> Agent 3."""
    def pipeline(io):
        state = dict(cfg)
        state.update(wiki_loader_agent(state))
        io.emit("log", text=f"Wiki loaded (code {state.get('assignment_code')}). Researching...")
        state.update(research_agent(state))
        io.emit("research", text=state.get("research_output", ""))
        io.emit("log", text="Research complete. Generating options...")
        state.update(problem_dataset_agent(state, io))

        if not state.get("approved") or not state.get("problem_statement"):
            io.emit("log", text="Stopped before evaluation (no approved problem statement).")
            return state

        io.emit("log", text="Problem approved. Building evaluation (Agent 3)...")
        state.update(evaluation_designer_agent(state, io))
        return state
    return pipeline


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


# ── session state ──────────────────────────────────────────────────────────
if "ws" not in st.session_state:
    st.session_state.ws = None

st.title("DSML Assignment Pipeline")
ws = st.session_state.ws

# ── START FORM ───────────────────────────────────────────────────────────
if ws is None:
    st.caption("Start a new assignment. config_type is fixed to vscode_type.")
    with st.form("start"):
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
            "config_type": "vscode_type",
            "assignment_code": code,
        }
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
        st.rerun()

# research brief (collapsible)
research = latest_event(events, "research")
if research and research.get("text"):
    with st.expander("Research brief", expanded=False):
        st.markdown(research["text"])

# recent notices
notices = [ev.get("text") for ev in events if ev.get("kind") == "notice"]
for note in notices[-2:]:
    st.info(note)

if status == "error":
    st.error(f"Pipeline error: {snap['error']}")
    st.stop()

if status in ("idle", "running"):
    st.spinner_text = "Working (research / dataset prep can take a minute)..."
    st.info("⏳ Working — research / dataset preparation can take a minute...")
    time.sleep(0.6)
    st.rerun()

if status == "done":
    res = ws.result or {}
    if res.get("evaluation_complete"):
        st.success("Pipeline complete — assignment generated.")
    else:
        st.warning("Stopped after Agent 2 (no approved problem statement).")

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
        add_noise, inject_nulls, drop_cols = 0.0, 0.0, []
        if atype == "tabular":
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
