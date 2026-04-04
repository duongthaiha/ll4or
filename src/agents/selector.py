"""Ensemble selector agent — picks the best answer from multiple solver results."""

from __future__ import annotations

import json
import logging
import re

from src.agents.base import Agent

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert operations research analyst. Multiple solver agents have \
attempted to solve the same optimization problem, producing different answers.

Given:
1. The original problem description.
2. The problem analysis (type, difficulty, structure).
3. Results from each solver (objective value, solver type, execution status).

Your task: select the SINGLE best answer. Consider:
- **Agreement**: If 2+ solvers agree (within 1%), prefer that value.
- **Solver fit**: For LP problems, heuristic (constructive) answers are often \
  more reliable. For combinatorial problems, metaheuristics may be better.
- **Outlier detection**: If one answer is wildly different from others, it's \
  likely wrong.
- **Feasibility**: A solver that ran successfully and converged is more \
  trustworthy than one that timed out or errored.
- **Objective direction**: For minimization, prefer the LOWEST feasible value. \
  For maximization, prefer the HIGHEST feasible value.

Return **only** valid JSON:
{
  "selected_solver": "<solver_type of the chosen result>",
  "selected_value": <number>,
  "confidence": <0.0 to 1.0>,
  "reasoning": "<brief explanation of selection logic>"
}
"""


def _build_user_prompt(
    question: str, analysis: dict, results: list[dict]
) -> str:
    analysis_str = json.dumps(analysis, indent=2, default=str)

    results_lines = []
    for r in results:
        comp = r.get("comparison")
        results_lines.append(
            f"  - {r.get('solver_type', '?')}: "
            f"obj={r.get('objective_value', 'N/A')}, "
            f"exec_success={r.get('execution_success', False)}, "
            f"elapsed={r.get('elapsed_seconds', 0):.1f}s"
        )

    return (
        f"## Problem Description\n{question}\n\n"
        f"## Problem Analysis\n```json\n{analysis_str}\n```\n\n"
        f"## Solver Results\n" + "\n".join(results_lines) + "\n\n"
        "Select the best answer and explain your reasoning."
    )


class SelectorAgent(Agent):
    """Selects the best answer from multiple solver results."""

    def run(self, input_data: dict) -> dict:
        question: str = input_data["question"]
        analysis: dict = input_data.get("analysis", {})
        results: list[dict] = input_data.get("results", [])

        user_prompt = _build_user_prompt(question, analysis, results)
        raw = self._chat(_SYSTEM_PROMPT, user_prompt, temperature=0.2)

        selection = self._parse_json(raw)
        return {
            **input_data,
            "selection": selection,
            "selection_raw": raw,
        }

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        candidate = match.group(1).strip() if match else text.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            log.warning("Failed to parse selector JSON")
            return {"selected_solver": "unknown", "confidence": 0.0}
