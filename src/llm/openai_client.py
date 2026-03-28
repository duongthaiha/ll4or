"""OpenAI LLM client implementation."""

from __future__ import annotations

from src.config import LLMConfig
from src.llm.base import LLMClient, Message


class OpenAIClient(LLMClient):
    """LLM client backed by the OpenAI API."""

    def __init__(self, config: LLMConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install openai") from exc

        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def chat(self, messages: list[Message], **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return response.choices[0].message.content or ""

    def generate(self, prompt: str, **kwargs) -> str:
        return self.chat([Message(role="user", content=prompt)], **kwargs)
