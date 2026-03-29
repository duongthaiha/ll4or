"""Azure OpenAI LLM client implementation."""

from __future__ import annotations

from src.config import LLMConfig
from src.llm.base import LLMClient, Message


class AzureOpenAIClient(LLMClient):
    """LLM client backed by Azure OpenAI Service.

    When Langfuse is active, uses the Langfuse-wrapped AzureOpenAI class
    so all API calls are automatically traced.
    """

    def __init__(self, config: LLMConfig):
        from src.tracing import get_traced_azure_client_class
        AzureOpenAI = get_traced_azure_client_class()

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
