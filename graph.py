"""
graph.py — the pipeline as a real LangGraph StateGraph.

This REPLACES the old hand-written `state.update(agent(state))` chain that used to
live in app.py. The agents are unchanged; here we wire them together as graph nodes
with LangGraph so the project genuinely runs on the framework it was always shaped for.

How LangGraph runs this
-----------------------
- `StateGraph(AssignmentState)` builds one channel per key in the state schema. When a
  node returns a dict, LangGraph merges those keys into the shared state (last write
  wins) — exactly what `state.update(...)` did by hand.
  IMPORTANT: LangGraph SILENTLY DROPS any returned key that is not declared in
  AssignmentState. That is why the schema must list every key the agents return.
- Edges define the flow. A conditional edge (a small `route_*` function) decides the
  next node from the current state — this replaces the old `if config_type == ...`.
- `graph.invoke(initial_state)` runs the nodes in order, in the CALLING thread. Our HITL
  agents pause by blocking on `io.ask(...)` (the WebSession thread+queue bridge), so
  invoke simply blocks at that node until the UI answers — same behavior as before.

Human-in-the-loop note
-----------------------
We keep the existing `io.ask` bridge INSIDE the nodes (Level 1 adoption). We are NOT
using LangGraph's native `interrupt()`/checkpoint resume yet — that is the deferred
Phase 4. So this graph is compiled WITHOUT a checkpointer and needs no thread_id.

The node wrappers below only add the same `io.emit(...)` log lines the old app.py
emitted, so the breadcrumb, research brief, and status text render identically.
"""

from langgraph.graph import StateGraph, START, END

from state import AssignmentState
from agents.wiki_loader import wiki_loader_agent
from agents.research import research_agent
from agents.problem_dataset import problem_dataset_agent
from agents.evaluation_designer import evaluation_designer_agent
from agents.codeeditor_design import codeeditor_design_agent
from agents.codeeditor_generate import codeeditor_generate_agent


def build_pipeline_graph(io):
    """Build and compile the pipeline graph. `io` is the run's WebSession/Terminal IO;
    the node functions close over it so they can emit logs and ask HITL questions.
    A fresh graph is built per run so each run's nodes talk to that run's `io`."""

    def _is_ce(state):
        return state.get("config_type") == "code_editor_type"

    # ── Nodes ──────────────────────────────────────────────────────────────
    # Each node calls the real agent and returns its state update. The only extra
    # code is the io.emit(...) lines, copied verbatim from the old app.py so the UI
    # logs are byte-for-byte the same.

    def n_wiki(state):
        upd = wiki_loader_agent(state)
        code = upd.get("assignment_code")
        if _is_ce(state):
            io.emit("log", text=f"Wiki loaded (code {code}).")
        else:
            io.emit("log", text=f"Wiki loaded (code {code}). Researching...")
        return upd

    def n_research(state):
        # Replicate the OLD per-flow gating EXACTLY (they differed):
        #   - vscode: research ran UNCONDITIONALLY (there was no use_research check).
        #   - code-editor: research ran only if use_research was truthy (absent -> skip),
        #     matching the old `if cfg.get("use_research")`.
        # This keeps behavior identical for every input, not just use_research=True.
        if _is_ce(state):
            if not state.get("use_research"):
                return {}
            io.emit("log", text="Researching code-editor sites for inspiration...")
        upd = research_agent(state)
        io.emit("research", text=upd.get("research_output", ""),
                sites=upd.get("research_sites", []))
        if not _is_ce(state):
            io.emit("log", text="Research complete. Generating options...")
        return upd

    def n_ce_design(state):
        io.emit("log", text="Designing the question...")
        return codeeditor_design_agent(state, io)

    def n_ce_generate(state):
        io.emit("log", text="Config approved. Generating deliverable...")
        return codeeditor_generate_agent(state, io)

    def n_vs_problem(state):
        return problem_dataset_agent(state, io)

    def n_vs_eval(state):
        io.emit("log", text="Problem approved. Building evaluation (Agent 3)...")
        return evaluation_designer_agent(state, io)

    # ── Routing (conditional edges) ─────────────────────────────────────────
    # These replace the old `if` statements. They may emit the same "Stopped ..."
    # log the old code printed at each early exit, then send the run to END.

    def route_by_type(state):
        return "ce_design" if _is_ce(state) else "vs_problem"

    def route_after_ce_design(state):
        if state.get("codeeditor_config"):
            return "ce_generate"
        io.emit("log", text="Stopped before generate (no approved config).")
        return END

    def route_after_vs_problem(state):
        if state.get("approved") and state.get("problem_statement"):
            return "vs_eval"
        io.emit("log", text="Stopped before evaluation (no approved problem statement).")
        return END

    # ── Assemble the graph ──────────────────────────────────────────────────
    g = StateGraph(AssignmentState)
    g.add_node("wiki", n_wiki)
    g.add_node("research", n_research)
    g.add_node("ce_design", n_ce_design)
    g.add_node("ce_generate", n_ce_generate)
    g.add_node("vs_problem", n_vs_problem)
    g.add_node("vs_eval", n_vs_eval)

    g.add_edge(START, "wiki")
    g.add_edge("wiki", "research")
    g.add_conditional_edges("research", route_by_type,
                            {"ce_design": "ce_design", "vs_problem": "vs_problem"})
    g.add_conditional_edges("ce_design", route_after_ce_design,
                            {"ce_generate": "ce_generate", END: END})
    g.add_edge("ce_generate", END)
    g.add_conditional_edges("vs_problem", route_after_vs_problem,
                            {"vs_eval": "vs_eval", END: END})
    g.add_edge("vs_eval", END)

    return g.compile()
