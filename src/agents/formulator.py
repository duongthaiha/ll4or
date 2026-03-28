"""Formulator agent — converts natural-language OR problem into structured formulation."""

from __future__ import annotations

import json
import logging
import re

from src.agents.base import Agent

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert operations research analyst. Given a natural-language \
optimization problem, extract a structured mathematical formulation.

Return **only** valid JSON with exactly these keys:
{
  "problem_type": "LP" | "IP" | "MIP" | "NLP" | "combinatorial" | "other",
  "objective": {
    "direction": "minimize" | "maximize",
    "description": "<what is being optimized>"
  },
  "decision_variables": [
    {"name": "<var>", "type": "continuous|integer|binary", "description": "<meaning>"}
  ],
  "constraints": [
    {"description": "<plain-English constraint>"}
  ],
  "parameters": [
    {"name": "<param>", "value": "<value or expression>"}
  ],
  "summary": "<one-sentence problem summary>"
}
"""


class FormulatorAgent(Agent):
    """Parses an OR problem description into a structured formulation."""

    def run(self, input_data: dict) -> dict:
        question: str = input_data["question"]
        raw = self._chat(_SYSTEM_PROMPT, question, temperature=0.2)

        formulation = self._parse_json(raw)
        return {
            **input_data,
            "formulation": formulation,
            "formulation_raw": raw,
        }

    @staticmethod
    def _parse_json(text: str) -> dict:
        # Try to extract JSON from markdown code blocks first
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        candidate = match.group(1).strip() if match else text.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            log.warning("Failed to parse formulation JSON, returning raw text")
            return {"raw": text}
