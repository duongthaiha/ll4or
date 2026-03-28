"""Debugger agent — fixes broken generated code using LLM-assisted repair."""

from __future__ import annotations

import logging

from src.agents.base import Agent

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert Python debugger.  A previously generated optimization \
solver script has failed during execution.

Given:
1. The original problem description.
2. The broken Python code.
3. The error message / traceback.

Your job is to fix the code so it runs successfully and produces the correct \
result.

RULES:
- Return the COMPLETE fixed Python script (not just the changed lines).
- Keep using ONLY Python standard library + numpy + scipy.
- The script must print its result in this exact format:
      OBJECTIVE_VALUE: <number>
- Do NOT change the overall algorithm approach — just fix the bugs.
- If the error is a logical error (wrong answer), you may adjust parameters \
  or fix the mathematical formulation.

Return ONLY the fixed Python code inside a single ```python ... ``` block.
"""


def _build_user_prompt(question: str, code: str, error: str) -> str:
    return (
        f"## Original Problem\n{question}\n\n"
        f"## Broken Code\n```python\n{code}\n```\n\n"
        f"## Error\n```\n{error}\n```\n\n"
        "Fix the code and return the complete corrected script."
    )


class DebuggerAgent(Agent):
    """Attempts to fix broken solver code using error feedback."""

    def run(self, input_data: dict) -> dict:
        question: str = input_data["question"]
        code: str = input_data["code"]
        error: str = input_data["error"]

        user_prompt = _build_user_prompt(question, code, error)
        raw = self._chat(_SYSTEM_PROMPT, user_prompt, temperature=0.2)

        return {
            **input_data,
            "fixed_code_raw": raw,
        }
