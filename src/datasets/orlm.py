"""ORLM / IndustryOR dataset adapter.

Reads the local JSON files that use the standard ORLM schema:
  { "en_question": str, "en_answer": float, ... }
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import EvaluationConfig
from src.datasets.base import DatasetAdapter, Problem

# Default data paths relative to repository root
_REPO_ROOT = Path(__file__).resolve().parents[2]

_DATASET_PATHS: dict[str, Path] = {
    "industryOR": _REPO_ROOT / "datasets" / "IndustryOR" / "IndustryOR.json",
    "nl4opt": _REPO_ROOT / "datasets" / "BWOR" / "data" / "datasets" / "NL4OPT_with_optimal_solution.json",
}


class ORLMAdapter(DatasetAdapter):
    """Adapter for datasets using the ORLM en_question/en_answer format."""

    def __init__(self, dataset_key: str = "industryOR", path: Path | None = None):
        self._key = dataset_key
        self._path = path or _DATASET_PATHS.get(dataset_key)
        if self._path is None:
            raise ValueError(
                f"Unknown dataset key '{dataset_key}'. "
                f"Available: {list(_DATASET_PATHS)}"
            )
        self._problems: list[Problem] = []

    @property
    def name(self) -> str:
        return f"orlm_{self._key}"

    def load(self) -> None:
        raw: list[dict] = []
        with open(self._path, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
            # Support both JSON array and JSONL
            if content.startswith("["):
                raw = json.loads(content)
            else:
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        raw.append(json.loads(line))

        self._problems = []
        for idx, entry in enumerate(raw):
            pid = str(entry.get("id", idx))
            question = entry.get("en_question", "")
            answer_raw = entry.get("en_answer")
            try:
                answer = float(answer_raw) if answer_raw is not None else None
            except (ValueError, TypeError):
                answer = None  # e.g. "No Best Solution"
            metadata = {
                k: v
                for k, v in entry.items()
                if k not in ("en_question", "en_answer")
            }
            metadata["raw_answer"] = str(answer_raw) if answer_raw is not None else None
            self._problems.append(
                Problem(id=pid, question=question, answer=answer, metadata=metadata)
            )

    def get_problems(self) -> list[Problem]:
        return list(self._problems)

    def get_eval_config(self) -> EvaluationConfig:
        """ORLM: 5% relative tolerance, round to integer, handle 'No Best Solution'."""
        return EvaluationConfig(
            comparison_mode="relative",
            relative_tolerance=0.05,
            absolute_tolerance=0.05,
            round_to_int=True,
            infeasible_values=["No Best Solution"],
        )
