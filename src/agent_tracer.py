"""Agent interaction tracer — captures agent inputs/outputs for debugging.

Enabled via AGENT_TRACE_DIR env var. When set, each agent's LLM call is
recorded with system prompt, user message, and response to a JSONL file.
Each agent's run() input_data and return value are also captured.

Example:
    AGENT_TRACE_DIR=./traces python3 -m src.main ...

Produces traces/problem_<id>.jsonl with one JSON record per agent call.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_trace_dir: Path | None = None
_current_problem_id: str | None = None


def init_from_env() -> None:
    """Initialize the tracer from AGENT_TRACE_DIR env var."""
    global _trace_dir
    path = os.environ.get("AGENT_TRACE_DIR")
    if path:
        _trace_dir = Path(path)
        _trace_dir.mkdir(parents=True, exist_ok=True)


def is_enabled() -> bool:
    return _trace_dir is not None


def set_problem(problem_id: str) -> None:
    """Set the current problem ID (used to route traces to per-problem files)."""
    global _current_problem_id
    _current_problem_id = str(problem_id)


def _trace_file() -> Path | None:
    if _trace_dir is None:
        return None
    pid = _current_problem_id or "unknown"
    return _trace_dir / f"problem_{pid}.jsonl"


def _write(record: dict) -> None:
    path = _trace_file()
    if path is None:
        return
    with _lock:
        with path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def record_llm_call(
    agent_name: str,
    system: str,
    user: str,
    response: str,
    kwargs: dict,
) -> None:
    """Record a single LLM call made by an agent."""
    _write({
        "timestamp": time.time(),
        "type": "llm_call",
        "agent": agent_name,
        "system_prompt": system,
        "user_prompt": user,
        "response": response,
        "kwargs": {k: v for k, v in kwargs.items() if k != "messages"},
    })


def record_agent_run(
    agent_name: str,
    input_data: dict,
    output_data: dict,
    elapsed_seconds: float,
) -> None:
    """Record an agent's full run() invocation with input and output."""
    # Filter out large/redundant keys for readability
    def _clean(d: dict) -> dict:
        return {
            k: v for k, v in d.items()
            if k not in ("answer",)  # skip ground truth to avoid leaking it visually
        }
    _write({
        "timestamp": time.time(),
        "type": "agent_run",
        "agent": agent_name,
        "elapsed_seconds": elapsed_seconds,
        "input": _clean(input_data),
        "output": _clean(output_data),
    })
