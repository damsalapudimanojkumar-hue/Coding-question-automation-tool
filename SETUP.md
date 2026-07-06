# Setup Guide — Step 0 & Step 1

## What's in this zip right now

```
assignment_pipeline_real/
├── state.py                  ← shared state shape (TypedDict) for the LangGraph pipeline
├── claude_client.py          ← wrapper for Claude API calls (text, search, tool-use loop)
├── test_connection.py        ← STEP 1: confirms your API key works
├── test_wiki_loader.py       ← STEP 2: confirms the wiki folder loads correctly
├── requirements.txt
├── .env.example
├── .gitignore
├── knowledge/                       ← the "wiki" — organized, filterable context
│   ├── assignments/
│   │   ├── vscode_type/             ← 15 past assignments, one .md each
│   │   └── code_editor_type/        ← empty for now, new category
│   ├── skills/                      ← ide-evaluation skill content
│   │   ├── ide_evaluation_skill.md
│   │   ├── detection_patterns.md
│   │   └── platform_rules.md
│   ├── instructions/
│   │   └── project_custom_instructions.md
│   └── dataset_library.json
├── tools/
│   └── local_tools.py        ← real code execution tools (run_python, run_pytest, write_file, read_file, list_dir)
├── agents/
│   └── wiki_loader.py        ← AGENT 0: reads knowledge/, assembles filtered context per agent
└── prompts/                  ← empty for now, built alongside each agent as we go
```

This is intentionally incomplete — Agents 1-5 (Research, Problem+Dataset
Designer, Evaluation Designer, Verification Loop, Packager) come next,
one at a time.

---

## Step 0 — Open in VSCode

1. Unzip this into a folder, open that folder in VSCode.
2. Open a terminal inside VSCode (`` Ctrl+` ``).

## Create a virtual environment (recommended)

```bash
python3 -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Add your API key

```bash
cp .env.example .env
```

Open `.env` in VSCode and replace `your_api_key_here` with your real key
(the one provided by your office team for API access):

```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Step 1 — Test the connection

```bash
python test_connection.py
```

**Expected output:**

```
Testing connection to Claude API...

Response received:
──────────────────────────────────────────────
Connection successful. Pipeline ready.
──────────────────────────────────────────────

If you see a response above, your setup is working.
Paste this output back to continue to Step 2.
```

If you get an error instead, common fixes:

| Error | Fix |
|---|---|
| `ANTHROPIC_API_KEY not found` | Check `.env` exists and has the key, no quotes needed |
| `ModuleNotFoundError: anthropic` | Run `pip install -r requirements.txt` again, check venv is activated |
| `401 Unauthorized` | API key is wrong or expired — check with your office team |
| `model not found` | We're using `claude-sonnet-4-6` — confirm your API access includes this model name |

---

## What to do next

Run `test_connection.py`, paste me the exact output (success or error),
and we'll move to Step 2 — testing the Wiki Loader.

---

## Step 2 — Test the Wiki Loader

No API key needed for this step — it's pure file reading and filtering,
no LLM calls.

```bash
python test_wiki_loader.py
```

This confirms the `knowledge/` folder structure is being read correctly:

```
knowledge/
├── assignments/
│   ├── vscode_type/        ← 15 past assignments as .md files
│   └── code_editor_type/   ← empty for now (README placeholder)
├── skills/
│   ├── ide_evaluation_skill.md
│   ├── detection_patterns.md
│   └── platform_rules.md
├── instructions/
│   └── project_custom_instructions.md
└── dataset_library.json
```

**Expected output:** four sections of real content printed (research
context, skill context, instructions context, dataset context), plus a
console summary like:

```
📚  AGENT 0: WIKI LOADER
Loading context for: tabular / vscode_type
  Past assignments loaded : 15 files
  Skill docs loaded       : 3 files
  Instructions loaded     : 1 files (2605 chars)
  Dataset library entries : 11
```

Paste me this output, and we move to Step 3 — building the real Research
agent (web search + this wiki context, filtered by assignment_type).

