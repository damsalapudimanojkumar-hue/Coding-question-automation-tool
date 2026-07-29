"""
tools/web_research.py

Domain-scoped web search for the Research agent, via Tavily (Way 1: a tool the
model calls). The model decides queries; we run them scoped to ML learning/
assignment sites and return snippets, so Research draws INSPIRATION (the
pattern/soul), never copies.

Scoped to vscode/notebook-format sites for now (Kaggle + data-science). The
code-editor sites (deep-ml, tensortonic, leetcode) get added when that mode
exists — keep them out here so a notebook run doesn't get off-format results.

Gated behind TAVILY_API_KEY: with no key (or SDK missing) the tool returns a
clear "unavailable" string and the Research agent falls back to its old search.
"""

import copy
import os
import re
from dotenv import load_dotenv

load_dotenv()

from tracing import observe

# vscode / notebook-format inspiration sites (Kaggle + data-science).
VSCODE_SITES = [
    "kaggle.com",
    "drivendata.org",
    "analyticsvidhya.com",
    "machinelearningmastery.com",
    "huggingface.co",
]

# code-editor / function-from-scratch inspiration sites. Deep-ML and TensorTonic
# are the bullseye (implement an ML function, graded by test cases); StrataScratch
# adds DS coding-interview framing; the last two give from-scratch algorithm
# explanations and difficulty calibration.
CODEEDITOR_SITES = [
    "deep-ml.com",
    "tensortonic.com",
    "stratascratch.com",
    "machinelearningmastery.com",
    "geeksforgeeks.org",
]

# Back-compat alias (older imports referenced TAVILY_SITES; it means "vscode").
TAVILY_SITES = VSCODE_SITES


def sites_for(config_type: str) -> list:
    """Pick the inspiration domain list for the assignment format."""
    return CODEEDITOR_SITES if config_type == "code_editor_type" else VSCODE_SITES


_client = None


def tavily_available() -> bool:
    load_dotenv()  # re-read .env so a key added after the app started is picked up
    return bool(os.getenv("TAVILY_API_KEY"))


def _get_client():
    """Lazily build the Tavily client; return None if unavailable."""
    global _client
    if _client is None:
        load_dotenv()
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            return None
        try:
            from tavily import TavilyClient
            _client = TavilyClient(api_key=key)
        except Exception:
            return None
    return _client


TAVILY_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "tavily_search",
        "description": (
            "Search ML learning and assignment sites (Kaggle and data-science blogs) for how "
            "a topic is taught and assessed. Use it to find good existing questions, datasets, "
            "and framings to draw INSPIRATION from - extract the underlying pattern/soul and "
            "the difficulty bar; never copy a question or reuse its dataset verbatim. Returns "
            "result titles, URLs, and content snippets. Search a few focused queries, refining "
            "based on what you find."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Focused search query"},
                "max_results": {
                    "type": "integer",
                    "description": "How many results (1-5), default 5",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def domains_from_urls(urls: list) -> list:
    """Reduce result URLs to a de-duplicated, order-preserving list of host domains."""
    doms = []
    for u in urls:
        m = re.match(r"https?://([^/]+)", u or "")
        if not m:
            continue
        d = m.group(1).lower()
        if d.startswith("www."):
            d = d[4:]
        if d and d not in doms:
            doms.append(d)
    return doms


@observe(as_type="tool")
def execute_tavily_search(tool_input: dict, sites: list = None, collector: list = None) -> str:
    client = _get_client()
    if client is None:
        return "[tavily_search unavailable: TAVILY_API_KEY not set]"

    query = (tool_input.get("query") or "").strip()
    if not query:
        return "[tavily_search error: empty query]"
    max_results = tool_input.get("max_results", 5) or 5
    try:
        max_results = max(1, min(int(max_results), 5))
    except (TypeError, ValueError):
        max_results = 5

    try:
        resp = client.search(
            query=query,
            include_domains=sites or VSCODE_SITES,
            search_depth="advanced",
            max_results=max_results,
        )
    except Exception as e:  # noqa: BLE001 - returned to the model, never crashes the run
        return f"[tavily_search error: {type(e).__name__}: {e}]"

    results = resp.get("results", []) if isinstance(resp, dict) else []
    if not results:
        return "[no results found on the scoped sites]"

    lines = []
    for r in results[:max_results]:
        title = (r.get("title") or "")[:120]
        url = r.get("url") or ""
        if collector is not None and url:
            collector.append(url)
        content = (r.get("content") or "")[:600]   # truncate to control token cost
        lines.append(f"- {title}\n  {url}\n  {content}")
    return "\n\n".join(lines)


def build_search_tool(config_type: str) -> dict:
    """Tool definition with a description tuned to the assignment format, so the
    model queries the right kind of source. Scoping is still enforced by
    include_domains in the executor."""
    tool = copy.deepcopy(TAVILY_SEARCH_TOOL)
    if config_type == "code_editor_type":
        tool["function"]["description"] = (
            "Search coding-practice and ML-from-scratch sites (Deep-ML, TensorTonic, "
            "StrataScratch, and from-scratch tutorials) for how a concept becomes a "
            "function-implementation question graded by test cases. Use it to see the "
            "pattern, the edge cases they test, and the difficulty bar - draw INSPIRATION, "
            "never copy a question verbatim. Returns titles, URLs, and content snippets. "
            "Run a few focused queries, refining based on what you find."
        )
    return tool


def make_tavily_executor(config_type: str):
    """Return a tool_executor closure bound to the domain list for this format. The
    returned callable exposes `.seen` — the list of result URLs surfaced this run —
    so the caller can report which sites were actually explored."""
    sites = sites_for(config_type)
    seen = []

    def _executor(tool_name: str, tool_input: dict) -> str:
        if tool_name == "tavily_search":
            return execute_tavily_search(tool_input, sites=sites, collector=seen)
        return f"[unknown tool: {tool_name}]"

    _executor.seen = seen
    return _executor


def tavily_tool_executor(tool_name: str, tool_input: dict) -> str:
    """Back-compat default (vscode-scoped) dispatch for call_claude_with_tools."""
    if tool_name == "tavily_search":
        return execute_tavily_search(tool_input, sites=VSCODE_SITES)
    return f"[unknown tool: {tool_name}]"
