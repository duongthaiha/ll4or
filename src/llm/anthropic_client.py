"""Anthropic (Claude) LLM client implementation."""

from __future__ import annotations

from src.config import LLMConfig
from src.llm.base import LLMClient, Message


class AnthropicClient(LLMClient):
    """LLM client backed by the Anthropic API."""

    def __init__(self, config: LLMConfig):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc

        self.config = config
        self._client = anthropic.Anthropic(api_key=config.api_key)

    def chat(self, messages: list[Message], **kwargs) -> str:
        # Anthropic requires system message to be passed separately
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        create_kwargs: dict = dict(
            model=kwargs.get("model", self.config.model),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            messages=chat_messages,
        )
        if system_msg:
            create_kwargs["system"] = system_msg

        response = self._client.messages.create(**create_kwargs)
        return response.content[0].text

    def generate(self, prompt: str, **kwargs) -> str:
        return self.chat([Message(role="user", content=prompt)], **kwargs)
