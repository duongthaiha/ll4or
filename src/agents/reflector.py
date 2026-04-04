"""Reflector agent — post-problem analysis for cross-problem learning."""

from __future__ import annotations

import json
import logging
import re

from src.agents.base import Agent

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert operations research meta-analyst. After a problem has been \
solved (or attempted), you analyze what happened to extract lessons for future \
problems.

Given:
1. The problem description and analysis.
2. Results from all solver attempts.
3. Whether the final answer was correct.
4. Accumulated lessons from previous problems in this run.

Extract concise, actionable lessons. Return **only** valid JSON:
{
  "lessons": [
    {
      "category": "formulation" | "algorithm_selection" | "parameter_tuning" | "data_handling" | "constraint_handling" | "convergence",
      "observation": "<what happened>",
      "recommendation": "<what to do differently for similar problems>"
    }
  ],
  "solver_performance": {
    "<solver_type>": {
      "worked": <bool>,
      "quality": "exact" | "close" | "far" | "failed",
      "notes": "<brief note>"
    }
  },
  "problem_pattern": "<one-line description of this problem type for future matching>"
}

Keep each lesson to 1-2 sentences. Focus on patterns, not problem-specific details.
"""


def _build_user_prompt(
    question: str,
    analysis: dict,
    results: list[dict],
    is_correct: bool,
    prior_lessons: list[dict],
) -> str:
    analysis_str = json.dumps(analysis, indent=2, default=str)

    results_lines = []
    for r in results:
        comp = r.get("comparison")
        correct = comp.is_correct if comp else False
        results_lines.append(
            f"  - {r.get('solver_type', '?')}: "
            f"obj={r.get('objective_value', 'N/A')}, "
            f"correct={correct}, "
            f"exec_success={r.get('execution_success', False)}"
        )

    prior_str = ""
    if prior_lessons:
        prior_str = (
            f"\n## Lessons From Previous Problems\n"
            f"```json\n{json.dumps(prior_lessons[-5:], indent=2)}\n```\n"
        )

    return (
        f"## Problem Description\n{question}\n\n"
        f"## Problem Analysis\n```json\n{analysis_str}\n```\n\n"
        f"## Solver Results\n" + "\n".join(results_lines) + "\n\n"
        f"## Final Result: {'CORRECT' if is_correct else 'INCORRECT'}\n"
        f"{prior_str}\n"
        "Extract lessons learned from this problem."
    )


class ReflectorAgent(Agent):
    """Extracts lessons from solved problems for cross-problem learning."""

    def run(self, input_data: dict) -> dict:
        question: str = input_data["question"]
        analysis: dict = input_data.get("analysis", {})
        results: list[dict] = input_data.get("results", [])
        is_correct: bool = input_data.get("is_correct", False)
        prior_lessons: list[dict] = input_data.get("prior_lessons", [])

        user_prompt = _build_user_prompt(
            question, analysis, results, is_correct, prior_lessons
        )
        raw = self._chat(_SYSTEM_PROMPT, user_prompt, temperature=0.3)

        reflection = self._parse_json(raw)
        return {
            **input_data,
            "reflection": reflection,
            "reflection_raw": raw,
        }

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        candidate = match.group(1).strip() if match else text.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            log.warning("Failed to parse reflector JSON")
            return {"lessons": [], "solver_performance": {}}
