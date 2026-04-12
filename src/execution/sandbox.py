"""Sandbox code executor — runs generated Python code in a subprocess."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.config import ExecutionConfig

log = logging.getLogger(__name__)

# Resolve the Python interpreter once at import time
_PYTHON = shutil.which("python3") or shutil.which("python") or sys.executable


@dataclass
class ExecutionResult:
    """Outcome of executing a solver script."""

    success: bool
    objective_value: float | None
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool


def extract_code(raw_response: str) -> str:
    """Pull Python code out of an LLM response (expects ```python blocks)."""
    # Try standard ```python ... ``` block (case-insensitive language tag)
    match = re.search(r"```[Pp](?:ython|y)?\s*\n(.*?)```", raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Handle truncated responses where closing ``` is missing
    match = re.search(r"```[Pp](?:ython|y)?\s*\n(.*)", raw_response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if code and ("import " in code or "def " in code or "print(" in code):
            log.debug("Extracted code from truncated code block (no closing ```)")
            return code
    # Fallback: if the response looks like pure code, use it directly
    if "def " in raw_response or "import " in raw_response:
        return raw_response.strip()
    return ""


def parse_objective_value(stdout: str) -> float | None:
    """Extract the OBJECTIVE_VALUE from script output."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        match = re.match(r"OBJECTIVE_VALUE:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
        if match:
            return float(match.group(1))
    return None


def execute_code(
    code: str,
    config: ExecutionConfig | None = None,
) -> ExecutionResult:
    """Execute Python code in a subprocess with timeout."""
    config = config or ExecutionConfig()

    if not code.strip():
        return ExecutionResult(
            success=False,
            objective_value=None,
            stdout="",
            stderr="No code to execute",
            return_code=-1,
            timed_out=False,
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=None
    ) as tmp:
        tmp.write(code)
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [_PYTHON, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=config.timeout,
            cwd=tmp_path.parent,
        )
        obj_val = parse_objective_value(result.stdout)
        return ExecutionResult(
            success=result.returncode == 0 and obj_val is not None,
            objective_value=obj_val,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            objective_value=None,
            stdout="",
            stderr=f"Execution timed out after {config.timeout}s",
            return_code=-1,
            timed_out=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
