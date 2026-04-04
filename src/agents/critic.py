"""Code critic agent — reviews generated solver code before execution."""

from __future__ import annotations

import json
import logging
import re

from src.agents.base import Agent

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert code reviewer specialising in optimization solver code.

Given:
1. The original problem description.
2. The mathematical formulation.
3. Generated Python solver code.
4. The solver type (heuristic, metaheuristic, or hyperheuristic).

Review the code for correctness BEFORE it is executed. Check for:

1. **Data correctness:** Are all numerical values from the problem correctly \
   embedded? Are there typos, wrong signs, or missing data?
2. **Constraint implementation:** Do the coded constraints match the formulation? \
   Are inequality directions correct (≤ vs ≥)?
3. **Objective function:** Is the optimization direction correct (min vs max)? \
   Is the objective formula correct?
4. **Numerical issues:** Division by zero, integer overflow, floating-point \
   precision problems, wrong data types.
5. **Output format:** Does the code end with `print(f"OBJECTIVE_VALUE: {value}")`?
6. **Convergence:** For metaheuristics — are iteration counts and parameters \
   reasonable? Will it converge within the time limit?
7. **Import restrictions:** Does it only use stdlib + numpy + scipy? \
   No gurobipy, coptpy, cplex, pyomo, or other commercial solvers.

Return **only** valid JSON:
{
  "approved": <bool>,
  "issues": [
    {
      "severity": "critical" | "warning" | "info",
      "category": "data" | "constraint" | "objective" | "numerical" | "output" | "convergence" | "import",
      "description": "<what is wrong>",
      "line_hint": "<approximate location or code snippet>",
      "fix": "<suggested fix>"
    }
  ],
  "overall_assessment": "<one-sentence summary>"
}

If the code looks correct, return {"approved": true, "issues": [], "overall_assessment": "Code appears correct."}.
Only flag REAL issues — do not flag style preferences or minor optimizations.
"""


def _build_user_prompt(
    question: str, formulation: dict, code: str, solver_type: str
) -> str:
    form_str = json.dumps(formulation, indent=2, default=str)
    return (
        f"## Problem Description\n{question}\n\n"
        f"## Mathematical Formulation\n```json\n{form_str}\n```\n\n"
        f"## Solver Type\n{solver_type}\n\n"
        f"## Generated Code\n```python\n{code}\n```\n\n"
        "Review this code for correctness. Only flag genuine issues."
    )


class CriticAgent(Agent):
    """Reviews solver code for correctness before execution."""

    def run(self, input_data: dict) -> dict:
        question: str = input_data["question"]
        formulation: dict = input_data.get("formulation", {})
        code: str = input_data["code"]
        solver_type: str = input_data.get("solver_type", "unknown")

        user_prompt = _build_user_prompt(question, formulation, code, solver_type)
        raw = self._chat(_SYSTEM_PROMPT, user_prompt, temperature=0.2)

        review = self._parse_json(raw)
        return {
            **input_data,
            "review": review,
            "review_raw": raw,
        }

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        candidate = match.group(1).strip() if match else text.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            log.warning("Failed to parse critic JSON, returning approved")
            return {"approved": True, "issues": [], "overall_assessment": "Parse failed"}
