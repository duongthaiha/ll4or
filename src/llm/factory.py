"""LLM client factory."""

from __future__ import annotations

from src.config import LLMConfig
from src.llm.base import LLMClient


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Instantiate the correct LLM client based on provider name."""
    if config.provider in ("openai", "ollama"):
        from src.llm.openai_client import OpenAIClient
        return OpenAIClient(config)
    elif config.provider == "anthropic":
        from src.llm.anthropic_client import AnthropicClient
        return AnthropicClient(config)
    elif config.provider == "azure":
        from src.llm.azure_client import AzureOpenAIClient
        return AzureOpenAIClient(config)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")
