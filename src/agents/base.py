"""Abstract base class for LLM-powered agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from src.llm.base import LLMClient, Message

log = logging.getLogger(__name__)


class Agent(ABC):
    """Base class for all agents in the pipeline.

    Each agent wraps a single LLM call (or a small chain) that transforms
    structured input into structured output.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    @abstractmethod
    def run(self, input_data: dict) -> dict:
        """Execute the agent's task and return results."""
        ...

    # ── helpers ──────────────────────────────────────────────────────

    def _chat(self, system: str, user: str, **kwargs) -> str:
        """Convenience: system + user message → assistant response."""
        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]
        return self.llm.chat(messages, **kwargs)
