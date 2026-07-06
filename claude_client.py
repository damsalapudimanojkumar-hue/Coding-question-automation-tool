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
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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
    )

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
        )

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
        )

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