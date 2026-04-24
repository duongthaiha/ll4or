"""Hyper-heuristic coder agent — generates code that selects/combines low-level heuristics."""

from __future__ import annotations

import logging

from src.agents.base import Agent

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert programmer and operations research engineer specialising \
in hyper-heuristic frameworks.

A **hyper-heuristic** operates at a higher level than metaheuristics: it \
does NOT directly manipulate solutions.  Instead, it manages a pool of \
low-level heuristic operators and decides — at each step — which operator to \
apply, adapting its selection strategy based on feedback.

Given:
1. A natural-language optimization problem.
2. A structured mathematical formulation.

Write a **self-contained Python script** that:
1. Defines 3–5 low-level heuristic operators (e.g., greedy construction, \
   local search swap, perturbation, crossover-based repair).
2. Implements a **selection hyper-heuristic** that adaptively chooses which \
   operator to apply at each iteration, using one of these strategies:
   - Roulette-wheel selection with performance-based weights.
   - Reinforcement-learning style (reward operators that improve the solution).
   - Choice-function (recency × improvement × diversity).
3. Runs for a configurable number of iterations and returns the best solution.

RULES:
- Use ONLY Python standard library + numpy + scipy.  Do NOT use any \
  commercial solver (no gurobipy, no coptpy, no cplex, no pyomo).
- The script must be **complete and runnable** — include all data from the \
  problem statement as literals in the code.
- The script should finish in under 300 seconds.  Use the time for more iterations.
- Handle constraints via penalty functions or repair operators.
- At the very end, print the result on a line by itself in this exact format:
      OBJECTIVE_VALUE: <number>
  where <number> is the best objective value found.
- Include brief comments explaining the hyper-heuristic design.

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

    lit = format_research_block(research, focus="hyperheuristic")
    if lit:
        parts.append(lit)

    if analysis:
        rec = analysis.get("recommended_solvers", {})
        operators = rec.get("hyperheuristic_operators", [])
        if operators:
            parts.append(
                f"## Recommended Low-Level Operators\n"
                f"Consider using these operators: {', '.join(operators)}\n\n"
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
        "Write a hyper-heuristic Python solver for this problem.  "
        "Define multiple low-level operators and an adaptive selection mechanism."
    )

    return "".join(parts)


class HyperHeuristicCoderAgent(Agent):
    """Generates hyper-heuristic solver scripts that select/combine low-level heuristics."""

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
            "solver_type": "hyperheuristic",
            "generated_code_raw": raw,
        }
