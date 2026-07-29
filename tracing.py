"""
tracing.py

Thin Langfuse wrapper that is GATED ON KEYS: tracing turns on only when
LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (in .env locally or in
Streamlit Secrets on the deployed app). With no keys, `observe` is a no-op
decorator and `flush()` does nothing, so the pipeline runs exactly as before.

Exposes:
    observe   - decorator to trace a function as a span/trace
    flush()   - send buffered traces (call at the end of a run)
    ENABLED   - bool, whether tracing is active
"""

import os
from dotenv import load_dotenv

load_dotenv()

ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

if ENABLED:
    try:
        from langfuse import observe, get_client  # noqa: F401

        def flush():
            try:
                get_client().flush()
            except Exception:
                pass
    except Exception:
        ENABLED = False

if not ENABLED:
    def observe(*args, **kwargs):
        """No-op decorator supporting both @observe and @observe(...)."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def _decorator(fn):
            return fn
        return _decorator

    def flush():
        pass
