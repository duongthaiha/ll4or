"""Dataset registry — maps human-friendly names to adapter instances."""

from __future__ import annotations

from src.datasets.base import DatasetAdapter
from src.datasets.bwor import BWORAdapter
from src.datasets.mamo import MAMOAdapter
from src.datasets.orlm import ORLMAdapter


def _build_default_registry() -> dict[str, DatasetAdapter]:
    return {
        "industryOR": ORLMAdapter("industryOR"),
        "bwor": BWORAdapter(),
        "mamo_easy": MAMOAdapter("easy"),
        "mamo_complex": MAMOAdapter("complex"),
    }


_REGISTRY: dict[str, DatasetAdapter] | None = None


def _registry() -> dict[str, DatasetAdapter]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_default_registry()
    return _REGISTRY


def list_datasets() -> list[str]:
    """Return names of all registered datasets."""
    return list(_registry().keys())


def get_dataset(name: str) -> DatasetAdapter:
    """Look up a dataset adapter by name."""
    reg = _registry()
    if name not in reg:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {list(reg)}"
        )
    return reg[name]


def register_dataset(name: str, adapter: DatasetAdapter) -> None:
    """Add a custom dataset at runtime."""
    _registry()[name] = adapter
