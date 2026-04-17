"""Abstract base class for LLM-powered agents."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from src import agent_tracer
from src.llm.base import LLMClient, Message
from src.tracing import get_observe, update_observation

log = logging.getLogger(__name__)


def _strip_answer(d):
    """Return a copy of dict `d` without the ground-truth `answer` key."""
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if k != "answer"}


class Agent(ABC):
    """Base class for all agents in the pipeline.

    Each agent wraps a single LLM call (or a small chain) that transforms
    structured input into structured output.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "run" not in cls.__dict__:
            return

        original_run = cls.run
        agent_name = cls.__name__

        def inner_run(self, input_data, __orig=original_run, __name=agent_name):
            """Core agent execution with custom tracer + span metadata."""
            start = time.time()
            output = __orig(self, input_data)
            if agent_tracer.is_enabled():
                agent_tracer.record_agent_run(
                    agent_name=__name,
                    input_data=input_data if isinstance(input_data, dict) else {"_": input_data},
                    output_data=output if isinstance(output, dict) else {"_": output},
                    elapsed_seconds=time.time() - start,
                )
            # Surface agent input/output on the current Langfuse span
            update_observation(
                input=_strip_answer(input_data),
                output=_strip_answer(output),
                metadata={"agent": __name},
            )
            return output

        def traced_run(self, input_data, __inner=inner_run, __name=agent_name):
            # Resolve @observe lazily: Langfuse is initialized *after* agents
            # are imported, so we can't wrap at class-definition time.
            observe = get_observe()
            wrapped = observe(name=__name)(__inner)
            return wrapped(self, input_data)

        cls.run = traced_run

    @abstractmethod
    def run(self, input_data: dict) -> dict:
        """Execute the agent's task and return results."""
        ...

    def __call__(self, input_data: dict) -> dict:
        return self.run(input_data)

    # ── helpers ──────────────────────────────────────────────────────

    def _chat(self, system: str, user: str, **kwargs) -> str:
        """Convenience: system + user message → assistant response."""
        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]
        response = self.llm.chat(messages, **kwargs)
        if agent_tracer.is_enabled():
            agent_tracer.record_llm_call(
                agent_name=type(self).__name__,
                system=system,
                user=user,
                response=response,
                kwargs=kwargs,
            )
        return response
