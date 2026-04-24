"""Researcher agent — identifies canonical OR problem type and surfaces textbook approaches.

Hybrid knowledge source: LLM parametric OR knowledge is grounded (when possible) by
a small curated local KB at `src/knowledge/or_problems.json`. If the LLM's canonical
name matches a KB entry (by alias), KB values take precedence for classic algorithms
and references while LLM-provided details augment the dossier.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.agents.base import Agent

log = logging.getLogger(__name__)

_DEFAULT_KB_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "or_problems.json"

_SYSTEM_PROMPT = """\
You are an operations research literature specialist.

Given a natural-language optimization problem and a preliminary classification, \
identify the canonical OR problem family this instance belongs to and summarize \
the textbook-standard approaches for that family.

Return **only** valid JSON with exactly these keys:
{
  "canonical_name": "<well-known OR problem name, e.g., 0/1 Knapsack Problem, Traveling Salesman Problem, Job-Shop Scheduling>",
  "problem_family": "<short category, e.g., combinatorial packing / network flow / scheduling / continuous nonlinear>",
  "complexity": "<P | NP-hard | NP-complete, with optional notes>",
  "exact_methods":         ["<classic exact algorithms or MILP formulations>"],
  "heuristic_methods":     ["<classic constructive / greedy heuristics with names>"],
  "metaheuristic_methods": ["<classic metaheuristics with names, e.g., GA with OX crossover, Lin-Kernighan, ALNS>"],
  "typical_parameters":    {"<param>": <value>},
  "key_pitfalls":          ["<common implementation mistakes to avoid>"],
  "references":            ["<textbook or seminal paper names>"]
}

Keep each list to 1-4 short entries. Use well-known algorithm names (e.g., \
"Hungarian algorithm", "Clarke-Wright savings", "NEH heuristic", "DSATUR"). \
If the problem is clearly linear and convex, recommend exact solvers (simplex / \
MILP / cvxpy) — do NOT suggest metaheuristics for polynomial-time problems.
"""


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def format_research_block(research: dict | None, *, focus: str = "heuristic") -> str:
    """Return a concise markdown block referencing the research dossier.

    `focus` ∈ {"heuristic", "metaheuristic", "hyperheuristic", "debug"} selects
    which method list to surface. Always short (≤ ~10 lines) to avoid crowding
    out problem-specific reasoning on small local models.
    """
    if not research or not isinstance(research, dict):
        return ""
    name = research.get("canonical_name") or ""
    if not name:
        return ""

    if focus == "metaheuristic":
        methods = research.get("metaheuristic_methods") or []
        label = "Recommended metaheuristic approach(es)"
    elif focus == "hyperheuristic":
        methods = (
            research.get("metaheuristic_methods")
            or research.get("heuristic_methods")
            or []
        )
        label = "Reference algorithms to draw low-level operators from"
    elif focus == "debug":
        methods = []
        label = ""
    else:  # heuristic
        methods = research.get("heuristic_methods") or []
        label = "Recommended heuristic approach(es)"

    pitfalls = research.get("key_pitfalls") or []
    params = research.get("typical_parameters") or {}

    lines = [f"## Literature Reference", f"Canonical problem: **{name}**"]
    if methods and label:
        top = methods[:2]
        lines.append(f"{label}: {'; '.join(top)}")
    if params and focus in ("metaheuristic", "hyperheuristic"):
        import json as _json
        lines.append(f"Typical parameters: {_json.dumps(params, default=str)}")
    if pitfalls:
        lines.append("Pitfalls to avoid:")
        for p in pitfalls[:3]:
            lines.append(f"  - {p}")
    return "\n".join(lines) + "\n\n"


class ResearcherAgent(Agent):
    """Identifies canonical OR problem type and retrieves reference approaches."""

    def __init__(self, llm, kb_path: str | Path | None = None):
        super().__init__(llm)
        self._kb_path = Path(kb_path) if kb_path else _DEFAULT_KB_PATH
        self._kb = self._load_kb(self._kb_path)
        self._alias_index = self._build_alias_index(self._kb)

    # ── public ──────────────────────────────────────────────────────

    def run(self, input_data: dict) -> dict:
        question: str = input_data["question"]
        analysis: dict = input_data.get("analysis") or {}

        user = self._build_user_prompt(question, analysis)
        raw = self._chat(_SYSTEM_PROMPT, user, temperature=0.2)
        research = self._parse_json(raw)

        # Ground the LLM's dossier against the curated KB when possible.
        matched_key = self._match_kb(research, analysis)
        if matched_key:
            research = self._merge_kb(research, self._kb[matched_key])
            research["_kb_match"] = matched_key

        return {
            **input_data,
            "research": research,
            "research_raw": raw,
        }

    # ── internals ────────────────────────────────────────────────────

    @staticmethod
    def _load_kb(path: Path) -> dict[str, dict]:
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            log.warning("Researcher KB not found at %s — running LLM-only", path)
            return {}
        # Strip meta entries
        return {k: v for k, v in data.items() if not k.startswith("_")}

    @staticmethod
    def _build_alias_index(kb: dict[str, dict]) -> dict[str, str]:
        idx: dict[str, str] = {}
        for key, entry in kb.items():
            idx[_normalize(key)] = key
            idx[_normalize(entry.get("canonical_name", ""))] = key
            for alias in entry.get("aliases", []):
                idx[_normalize(alias)] = key
        idx.pop("", None)
        return idx

    @staticmethod
    def _build_user_prompt(question: str, analysis: dict) -> str:
        parts = [f"## Problem\n{question}\n"]
        if analysis:
            hint_bits = []
            pc = analysis.get("problem_class")
            if pc:
                hint_bits.append(f"preliminary problem_class: {pc}")
            ks = (analysis.get("structure") or {}).get("known_structure")
            if ks:
                hint_bits.append(f"known_structure: {ks}")
            if hint_bits:
                parts.append(f"\n## Analyzer hint\n{'; '.join(hint_bits)}\n")
        parts.append(
            "\nIdentify the canonical OR problem family and return the JSON dossier."
        )
        return "".join(parts)

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        candidate = match.group(1).strip() if match else text.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            log.warning("Researcher: failed to parse JSON, returning raw text")
            return {"raw": text}

    def _match_kb(self, research: dict, analysis: dict) -> str | None:
        """Try to match the research dossier to a KB entry by canonical name or aliases."""
        candidates: list[str] = []
        cn = research.get("canonical_name")
        if isinstance(cn, str):
            candidates.append(cn)
        pf = research.get("problem_family")
        if isinstance(pf, str):
            candidates.append(pf)
        # Fall back to analyzer hints
        pc = analysis.get("problem_class")
        if isinstance(pc, str):
            candidates.append(pc)
        ks = (analysis.get("structure") or {}).get("known_structure")
        if isinstance(ks, str):
            candidates.append(ks)

        for cand in candidates:
            key = self._alias_index.get(_normalize(cand))
            if key:
                return key
            # Token-level contains match (e.g., "0/1 knapsack problem" → "knapsack")
            toks = _normalize(cand).split()
            for alias_norm, kb_key in self._alias_index.items():
                alias_toks = alias_norm.split()
                if alias_toks and all(t in toks for t in alias_toks):
                    return kb_key
        return None

    @staticmethod
    def _merge_kb(llm: dict, kb: dict) -> dict:
        """Merge KB entry into the LLM's dossier. KB takes precedence for grounded fields."""
        out = dict(llm) if isinstance(llm, dict) else {}
        grounded_fields = [
            "canonical_name",
            "complexity",
            "exact_methods",
            "heuristic_methods",
            "metaheuristic_methods",
            "typical_parameters",
            "key_pitfalls",
            "references",
        ]
        for f in grounded_fields:
            if f in kb:
                out[f] = kb[f]
        # Preserve problem_family from LLM if present, else infer
        out.setdefault("problem_family", llm.get("problem_family", ""))
        return out
