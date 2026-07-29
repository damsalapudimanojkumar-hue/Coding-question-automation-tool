"""
claude_client.py

Single wrapper for all model API calls in this pipeline.
Every agent uses call_claude() / call_claude_with_search() /
call_claude_with_tools() — keeps API logic in one place.

CHANGED: now routes through OpenRouter instead of calling Anthropic
directly. OpenRouter exposes an OpenAI-compatible endpoint, so we use
the `openai` SDK (not `anthropic`) pointed at OpenRouter's base_url.
Model names now need the "anthropic/" prefix (OpenRouter's naming
convention), and tool/response shapes follow OpenAI's schema, not
Anthropic's — see the request/response handling below, it's different
from before.
"""

import os
import threading
from dotenv import load_dotenv

load_dotenv()

# ── Per-run cost tracking (OpenRouter "usage accounting") ──────────────────
# We ask OpenRouter to return the ACTUAL cost it charged per call (via
# extra_body usage.include) and accumulate it here. Single run at a time, so a
# module-level tracker with a lock is enough. reset_cost() at the start of a
# run; cost_summary() to read the total (also works for failed/partial runs).

_USAGE_ACCOUNTING = {"usage": {"include": True}}


class _CostTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.cost = 0.0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.calls = 0

    def record(self, usage):
        # A call happened regardless of whether the provider returned usage data,
        # so always count it; only add tokens/cost when usage is present.
        with self._lock:
            self.calls += 1
            if usage is None:
                return
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.cost += _extract_cost(usage)

    def summary(self):
        with self._lock:
            return {
                "cost_usd": round(self.cost, 6),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "calls": self.calls,
            }


def _extract_cost(usage) -> float:
    """OpenRouter returns the real charged cost as `usage.cost`. The OpenAI SDK
    may expose it directly or tuck it into model_extra — handle both."""
    cost = getattr(usage, "cost", None)
    if cost is None:
        extra = getattr(usage, "model_extra", None) or {}
        cost = extra.get("cost")
    try:
        return float(cost) if cost is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


_tracker = _CostTracker()


def reset_cost():
    """Clear the tally at the start of a run."""
    _tracker.reset()


def cost_summary() -> dict:
    """Totals for the current run: cost_usd, tokens, calls."""
    return _tracker.summary()

# Use the Langfuse-traced OpenAI client when Langfuse keys are present; otherwise
# the plain client. Import-gated so nothing changes until keys are configured.
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    try:
        from langfuse.openai import OpenAI   # drop-in, auto-traces every call
    except Exception:
        from openai import OpenAI
else:
    from openai import OpenAI

_client = None

# OpenRouter's slug for Claude Sonnet 4.6. If your office team's
# OpenRouter account uses a different alias, change this one line.
DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. "
                "Copy .env.example to .env and add your OpenRouter key."
            )
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client


def call_claude(
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> str:
    """
    Make a single model API call. Returns the text response as a string.
    """
    client = get_client()

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        extra_body=_USAGE_ACCOUNTING,
    )
    _tracker.record(getattr(response, "usage", None))

    return response.choices[0].message.content


def call_claude_with_search(
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    max_searches: int = 8,
    max_turns: int = 6,
) -> str:
    """
    Model call with real web search enabled via OpenRouter's
    `openrouter:web_search` server tool. The model decides if/when to
    search; OpenRouter runs the search server-side and feeds results
    back to the model, which may search multiple times before producing
    a final answer.

    Unlike calling Anthropic directly, OpenRouter's server tool still
    requires US to loop: the model returns a tool_call, we don't execute
    anything ourselves (OpenRouter already ran the search and will return
    results on the next call if we just continue the loop with the
    tool_call still pending) — in practice with server tools the search
    typically resolves within the same response, but we keep a small
    loop as a safety net in case a model requires an extra round-trip.
    """
    client = get_client()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    tools = [{
        "type": "openrouter:web_search",
        "parameters": {
            "engine": "native",       # use Anthropic's own search when available
            "max_results": 5,
            "max_total_results": max_searches,
        },
    }]

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
            extra_body=_USAGE_ACCOUNTING,
        )
        _tracker.record(getattr(response, "usage", None))

        choice = response.choices[0]
        message = choice.message

        if choice.finish_reason != "tool_calls":
            return (message.content or "").strip()

        # Server tool calls are executed by OpenRouter itself, not by us,
        # but we still need to append the assistant turn and continue the
        # loop so the model can read the results and keep going.
        messages.append(message.model_dump())

    return "[max_turns reached without a final answer]"


def call_claude_with_tools(
    system: str,
    user: str,
    tools: list,
    tool_executor,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    max_turns: int = 10,
) -> str:
    """
    Generic agentic loop for OUR OWN local tools (run_python, run_pytest,
    write_file, etc — see tools/local_tools.py). The model can call these
    repeatedly until it produces a final text answer.

    tools: list of tool definitions in OPENAI function-calling format
           (NOTE: this is a different schema than Anthropic's — see the
           "input_schema" -> "parameters" rename needed in local_tools.py,
           flagged separately).
    tool_executor: function(tool_name: str, tool_input: dict) -> str
                   you provide this — it actually runs the tool and
                   returns the result as a string.
    """
    client = get_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
            extra_body=_USAGE_ACCOUNTING,
        )
        _tracker.record(getattr(response, "usage", None))

        choice = response.choices[0]
        message = choice.message

        if choice.finish_reason != "tool_calls":
            return (message.content or "").strip()

        messages.append(message.model_dump())

        for tool_call in message.tool_calls:
            import json
            tool_name = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)
            print(f"  [tool call] {tool_name}({tool_input})")
            result = tool_executor(tool_name, tool_input)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)[:8000],
            })

    return "[max_turns reached without a final answer]"