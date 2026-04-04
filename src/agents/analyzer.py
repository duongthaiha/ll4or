"""Problem analyzer agent — classifies and routes OR problems for adaptive solving."""

from __future__ import annotations

import json
import logging
import re

from src.agents.base import Agent

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert operations research analyst specialising in problem \
classification and algorithm selection.

Given a natural-language optimization problem, analyze it and return a \
structured classification that will guide downstream solver agents.

Return **only** valid JSON with exactly these keys:
{
  "problem_class": "LP" | "IP" | "MIP" | "NLP" | "combinatorial" | "network_flow" | "scheduling" | "assignment" | "knapsack" | "TSP" | "VRP" | "other",
  "difficulty": "easy" | "medium" | "hard",
  "scale": {
    "num_variables_estimate": <int>,
    "num_constraints_estimate": <int>,
    "is_large_scale": <bool>
  },
  "structure": {
    "is_linear": <bool>,
    "has_integer_vars": <bool>,
    "has_binary_vars": <bool>,
    "has_nonlinear_obj": <bool>,
    "has_nonlinear_constraints": <bool>,
    "is_convex": <bool or null if unknown>,
    "known_structure": "<e.g., knapsack, bin_packing, TSP, assignment, flow, scheduling, or null>"
  },
  "recommended_solvers": {
    "heuristic_strategy": "<specific heuristic approach, e.g., greedy-by-ratio for knapsack, nearest-neighbor for TSP>",
    "metaheuristic_algorithm": "<best-fit algorithm: GA | SA | PSO | TabuSearch | DE | ALNS | ACO>",
    "metaheuristic_params": {
      "population_size": <int or null>,
      "iterations": <int>,
      "cooling_rate": <float or null>
    },
    "hyperheuristic_operators": ["<list of 3-5 recommended low-level operators>"]
  },
  "decomposition_possible": <bool>,
  "decomposition_hints": ["<list of sub-problem descriptions if decomposable, else empty>"],
  "key_challenges": ["<list of anticipated difficulties, e.g., tight constraints, large search space, multiple optima>"]
}
"""


class AnalyzerAgent(Agent):
    """Classifies an OR problem and recommends solver strategies."""

    def run(self, input_data: dict) -> dict:
        question: str = input_data["question"]
        raw = self._chat(_SYSTEM_PROMPT, question, temperature=0.2)

        analysis = self._parse_json(raw)
        return {
            **input_data,
            "analysis": analysis,
            "analysis_raw": raw,
        }

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        candidate = match.group(1).strip() if match else text.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            log.warning("Failed to parse analysis JSON, returning raw text")
            return {"raw": text}
