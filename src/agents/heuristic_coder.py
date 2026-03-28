"""Heuristic coder agent — generates problem-specific constructive/greedy heuristic code."""

from __future__ import annotations

import logging

from src.agents.base import Agent

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert programmer and operations research engineer.

Given:
1. A natural-language optimization problem.
2. A structured mathematical formulation of the problem.

Write a **self-contained Python script** that solves the problem using a \
**problem-specific heuristic** (greedy, constructive, rule-based, or \
domain-specific heuristic).

RULES:
- Use ONLY Python standard library + numpy + scipy.  Do NOT use any \
  commercial solver (no gurobipy, no coptpy, no cplex, no pyomo).
- The script must be **complete and runnable** — include all data from the \
  problem statement as literals in the code.
- At the very end, print the result on a line by itself in this exact format:
      OBJECTIVE_VALUE: <number>
  where <number> is the best objective value found.
- Include brief comments explaining the heuristic strategy.
- Handle edge cases gracefully (e.g., infeasible relaxations).

Return ONLY the Python code inside a single ```python ... ``` block.
"""


def _build_user_prompt(question: str, formulation: dict) -> str:
    import json

    form_str = json.dumps(formulation, indent=2, default=str)
    return (
        f"## Problem Description\n{question}\n\n"
        f"## Mathematical Formulation\n```json\n{form_str}\n```\n\n"
        "Write a heuristic Python solver for this problem."
    )


class HeuristicCoderAgent(Agent):
    """Generates a greedy/constructive heuristic solver script."""

    def run(self, input_data: dict) -> dict:
        question: str = input_data["question"]
        formulation: dict = input_data.get("formulation", {})

        user_prompt = _build_user_prompt(question, formulation)
        raw = self._chat(_SYSTEM_PROMPT, user_prompt)

        return {
            **input_data,
            "solver_type": "heuristic",
            "generated_code_raw": raw,
        }
