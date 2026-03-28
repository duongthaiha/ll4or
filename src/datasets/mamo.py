"""MAMO dataset adapter.

Reads datasets/MAMO/HF_data/MAMO_EasyLP.json and MAMO_ComplexLP.json.
Schema: { "en_question": str, "en_answer": float }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from src.config import EvaluationConfig
from src.datasets.base import DatasetAdapter, Problem

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HF_DIR = _REPO_ROOT / "datasets" / "MAMO" / "HF_data"

_SPLIT_FILES = {
    "easy": _HF_DIR / "MAMO_EasyLP.json",
    "complex": _HF_DIR / "MAMO_ComplexLP.json",
}


class MAMOAdapter(DatasetAdapter):
    """Adapter for MAMO Easy/Complex LP benchmarks."""

    def __init__(
        self,
        split: Literal["easy", "complex"] = "easy",
        path: Path | None = None,
    ):
        self._split = split
        self._path = path or _SPLIT_FILES.get(split)
        if self._path is None:
            raise ValueError(f"Unknown split '{split}'. Available: {list(_SPLIT_FILES)}")
        self._problems: list[Problem] = []

    @property
    def name(self) -> str:
        return f"mamo_{self._split}"

    def load(self) -> None:
        raw: list[dict] = []
        with open(self._path, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
            if content.startswith("["):
                raw = json.loads(content)
            else:
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        raw.append(json.loads(line))

        self._problems = [
            Problem(
                id=str(idx),
                question=entry["en_question"],
                answer=float(entry["en_answer"]) if entry.get("en_answer") is not None else None,
                metadata={"split": self._split},
            )
            for idx, entry in enumerate(raw)
        ]

    def get_problems(self) -> list[Problem]:
        return list(self._problems)

    def get_eval_config(self) -> EvaluationConfig:
        """MAMO: hybrid comparison — scale-based decimal check OR 0.01% relative."""
        return EvaluationConfig(
            comparison_mode="mamo_hybrid",
            relative_tolerance=1e-4,
            round_to_int=False,
        )
