"""Azure OpenAI LLM client implementation."""

from __future__ import annotations

from src.config import LLMConfig
from src.llm.base import LLMClient, Message


class AzureOpenAIClient(LLMClient):
    """LLM client backed by Azure OpenAI Service."""

    def __init__(self, config: LLMConfig):
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise ImportError("pip install openai") from exc

        self.config = config
        self._client = AzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.base_url or "",
            api_version=config.api_version or "2025-04-01-preview",
        )

    def chat(self, messages: list[Message], **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=kwargs.get("temperature", self.config.temperature),
            max_completion_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return response.choices[0].message.content or ""

    def generate(self, prompt: str, **kwargs) -> str:
        return self.chat([Message(role="user", content=prompt)], **kwargs)
