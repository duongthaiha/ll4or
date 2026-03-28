"""Abstract dataset adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.config import EvaluationConfig


@dataclass
class Problem:
    """A single OR problem extracted from a dataset."""

    id: str
    question: str
    answer: float | None
    metadata: dict


class DatasetAdapter(ABC):
    """Interface for loading OR problem datasets.

    Subclasses know how to read a specific dataset format and expose
    a uniform list of Problem objects.  The solver pipeline only sees
    Problem — it never touches the raw dataset files.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this dataset (e.g. 'orlm_industryOR')."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load/parse the dataset from disk or network."""
        ...

    @abstractmethod
    def get_problems(self) -> list[Problem]:
        """Return all problems after load() has been called."""
        ...

    def get_eval_config(self) -> EvaluationConfig:
        """Return dataset-specific evaluation settings.

        Override in subclasses to match the benchmark's reference evaluation.
        Default: 5% relative tolerance (ORLM-style).
        """
        return EvaluationConfig()
