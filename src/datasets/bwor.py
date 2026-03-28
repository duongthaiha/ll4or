"""BWOR dataset adapter.

Reads datasets/BWOR/data/datasets/BWOR.json  (82 entries, JSONL format).
Schema: { "en_question": str, "en_answer": float, "difficulty": str, "id": int }
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import EvaluationConfig
from src.datasets.base import DatasetAdapter, Problem

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _REPO_ROOT / "datasets" / "BWOR" / "data" / "datasets" / "BWOR.json"


class BWORAdapter(DatasetAdapter):
    """Adapter for the BWOR benchmark."""

    def __init__(self, path: Path | None = None):
        self._path = path or _DEFAULT_PATH
        self._problems: list[Problem] = []

    @property
    def name(self) -> str:
        return "bwor"

    def load(self) -> None:
        raw: list[dict] = []
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))

        self._problems = []
        for idx, entry in enumerate(raw):
            raw_answer = entry.get("en_answer")
            try:
                answer = float(raw_answer) if raw_answer is not None else None
            except (ValueError, TypeError):
                answer = None  # e.g. "No Best Solution"
            self._problems.append(
                Problem(
                    id=str(entry.get("id", idx)),
                    question=entry["en_question"],
                    answer=answer,
                    metadata={
                        "difficulty": entry.get("difficulty", ""),
                        "cn_question": entry.get("cn_question", ""),
                        "raw_answer": str(raw_answer) if raw_answer is not None else None,
                    },
                )
            )

    def get_problems(self) -> list[Problem]:
        return list(self._problems)

    def get_eval_config(self) -> EvaluationConfig:
        """BWOR: absolute error < 0.1, no rounding, None for infeasible."""
        return EvaluationConfig(
            comparison_mode="absolute",
            absolute_tolerance=0.1,
            round_to_int=False,
            infeasible_values=["None", "No Best Solution"],
        )
