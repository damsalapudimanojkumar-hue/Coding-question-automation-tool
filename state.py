from typing import TypedDict, Optional, List, Dict


class AssignmentState(TypedDict):
    """
    Shared state across the whole pipeline.
    Every node reads from this and returns updates to it.
    """

    # ── You provide at start ───────────────────────────────────────────────
    topic: str
    learning_objective: str
    assignment_type: str                  # "tabular" | "nlp" | "cv"
    config_type: str                      # "vscode" (notebook/exec) | "code_editor" (function/DSA-style)
    assignment_code: Optional[str]        # short code (SLR, REG, BAG...) suffixing data/gt/zip files

    # ── Agent 0: Wiki Loader ───────────────────────────────────────────────
    wiki_research_context: Optional[str]
    wiki_skill_context: Optional[str]
    wiki_instructions_context: Optional[str]
    wiki_dataset_context: Optional[str]
    wiki_reference_formats: Optional[str]
    wiki_examples: Optional[str]           # complete worked-example bundles
    output_dir: Optional[str]  # absolute assignment-specific output folder

    # ── Agent: Research ────────────────────────────────────────────────────
    research_output: Optional[str]

    # ── Agent 2: Problem + Dataset Designer (3-phase with HITL) ────────────
    # Phase 1: candidate options
    problem_dataset_options: Optional[List[Dict]]
    options_table: Optional[List[Dict]]
    # Phase 2: human selects one + optional edits
    selected_option_index: Optional[int]
    selected_option: Optional[Dict]
    problem_statement_draft: Optional[str]
    dataset_modification_notes: Optional[str]
    # Phase 3: final approved versions
    problem_statement: Optional[str]                # final approved
    dataset_plan: Optional[str]                     # final approved
    dataset_files: Optional[Dict]                   # {filename: path}
    dataset_files_path: Optional[str]

    # ── Agent: Test Case Designer ──────────────────────────────────────────
    test_cases: Optional[str]

    # ── Agent: Solution + File Generator (absorbs your existing Claude Project) ──
    generated_bundle: Optional[Dict]      # {filename: content} - solution.ipynb, conftest.py, etc.

    # ── Agent: Verification Loop ───────────────────────────────────────────
    verification_report: Optional[str]    # pytest results against reference + lazy solutions
    verification_passed: Optional[bool]

    # ── Agent: Packager ────────────────────────────────────────────────────
    final_zips: Optional[Dict]            # {zip_name: path}

    # ── Human feedback (reused at every checkpoint) ────────────────────────
    human_feedback: Optional[str]
    approved: Optional[bool]

    # ── Pipeline control ───────────────────────────────────────────────────
    current_stage: Optional[str]
    retry_count: Optional[int]
