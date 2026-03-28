"""Abstract LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Message:
    """A single chat message."""

    role: str  # "system", "user", "assistant"
    content: str


class LLMClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def chat(self, messages: list[Message], **kwargs) -> str:
        """Send a chat conversation and return the assistant's response."""
        ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a completion for a single prompt."""
        ...
