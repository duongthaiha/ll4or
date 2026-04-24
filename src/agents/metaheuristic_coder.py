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
  finishes in under 300 seconds.  Prefer MORE iterations for better quality.
- Handle constraints via penalty functions or repair operators.
- At the very end, print the result on a line by itself in this exact format:
      OBJECTIVE_VALUE: <number>
  where <number> is the best objective value found.
- Include brief comments explaining the chosen metaheuristic and its parameters.

Return ONLY the Python code inside a single ```python ... ``` block.
"""


def _build_user_prompt(
    question: str, formulation: dict, analysis: dict | None = None,
    warm_start: dict | None = None, research: dict | None = None,
) -> str:
    import json
    from src.agents.researcher import format_research_block

    form_str = json.dumps(formulation, indent=2, default=str)

    parts = [
        f"## Problem Description\n{question}\n\n"
        f"## Mathematical Formulation\n```json\n{form_str}\n```\n\n"
    ]

    lit = format_research_block(research, focus="metaheuristic")
    if lit:
        parts.append(lit)

    if analysis:
        rec = analysis.get("recommended_solvers", {})
        algo = rec.get("metaheuristic_algorithm", "")
        params = rec.get("metaheuristic_params", {})
        if algo:
            parts.append(
                f"## Recommended Algorithm\n"
                f"Use **{algo}** for this problem.\n"
            )
            if params:
                parts.append(
                    f"Suggested parameters: {json.dumps(params, default=str)}\n\n"
                )

    if warm_start and warm_start.get("objective_value") is not None:
        parts.append(
            f"## Reference From Heuristic (use with caution)\n"
            f"A constructive heuristic found objective value "
            f"**{warm_start['objective_value']}**, but this may be WRONG.\n"
            f"You may use it as a rough reference point, but you MUST "
            f"derive your own solution independently. Do NOT assume the "
            f"heuristic's answer is correct — verify by solving from scratch.\n\n"
        )

    parts.append(
        "Write a metaheuristic Python solver for this problem.  "
        "Choose the most suitable algorithm and explain your choice briefly in a comment."
    )

    return "".join(parts)


class MetaheuristicCoderAgent(Agent):
    """Generates metaheuristic solver scripts (GA, SA, PSO, etc.)."""

    def run(self, input_data: dict) -> dict:
        question: str = input_data["question"]
        formulation: dict = input_data.get("formulation", {})
        analysis: dict | None = input_data.get("analysis")
        warm_start: dict | None = input_data.get("warm_start")
        research: dict | None = input_data.get("research")

        user_prompt = _build_user_prompt(
            question, formulation, analysis=analysis,
            warm_start=warm_start, research=research,
        )
        raw = self._chat(_SYSTEM_PROMPT, user_prompt)

        return {
            **input_data,
            "solver_type": "metaheuristic",
            "generated_code_raw": raw,
        }
