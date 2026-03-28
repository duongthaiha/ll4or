"""Metaheuristic coder agent — generates GA, SA, PSO, Tabu Search solvers."""

from __future__ import annotations

import logging

from src.agents.base import Agent

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert programmer and operations research engineer specialising \
in metaheuristic algorithms.

Given:
1. A natural-language optimization problem.
2. A structured mathematical formulation.

Write a **self-contained Python script** that solves the problem using a \
**metaheuristic algorithm**.  Choose the most appropriate metaheuristic for \
the problem type:
- Genetic Algorithm (GA) for combinatorial / integer problems.
- Simulated Annealing (SA) for continuous / mixed-integer problems.
- Particle Swarm Optimization (PSO) for continuous problems.
- Tabu Search for discrete search spaces.
- Differential Evolution for nonlinear continuous problems.

RULES:
- Use ONLY Python standard library + numpy + scipy.  Do NOT use any \
  commercial solver (no gurobipy, no coptpy, no cplex, no pyomo).
- The script must be **complete and runnable** — include all data from the \
  problem statement as literals in the code.
- Use a reasonable number of iterations/population size so the script \
  finishes in under 60 seconds.
- Handle constraints via penalty functions or repair operators.
- At the very end, print the result on a line by itself in this exact format:
      OBJECTIVE_VALUE: <number>
  where <number> is the best objective value found.
- Include brief comments explaining the chosen metaheuristic and its parameters.

Return ONLY the Python code inside a single ```python ... ``` block.
"""


def _build_user_prompt(question: str, formulation: dict) -> str:
    import json

    form_str = json.dumps(formulation, indent=2, default=str)
    return (
        f"## Problem Description\n{question}\n\n"
        f"## Mathematical Formulation\n```json\n{form_str}\n```\n\n"
        "Write a metaheuristic Python solver for this problem.  "
        "Choose the most suitable algorithm and explain your choice briefly in a comment."
    )


class MetaheuristicCoderAgent(Agent):
    """Generates metaheuristic solver scripts (GA, SA, PSO, etc.)."""

    def run(self, input_data: dict) -> dict:
        question: str = input_data["question"]
        formulation: dict = input_data.get("formulation", {})

        user_prompt = _build_user_prompt(question, formulation)
        raw = self._chat(_SYSTEM_PROMPT, user_prompt)

        return {
            **input_data,
            "solver_type": "metaheuristic",
            "generated_code_raw": raw,
        }
