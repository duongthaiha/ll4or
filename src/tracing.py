"""Langfuse tracing integration.

Provides:
  - init_langfuse(config) — sets up Langfuse env vars and returns whether it's active
  - get_observe() — returns the @observe decorator (or a no-op if Langfuse is disabled)
  - get_traced_azure_client_class() — returns Langfuse-wrapped AzureOpenAI class
  - get_traced_openai_client_class() — returns Langfuse-wrapped OpenAI class
"""

from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Any, Callable

from src.config import LangfuseConfig

log = logging.getLogger(__name__)

_LANGFUSE_ACTIVE = False


def init_langfuse(config: LangfuseConfig) -> bool:
    """Initialize Langfuse via environment variables. Returns True if active."""
    global _LANGFUSE_ACTIVE

    if not config.enabled:
        log.info("Langfuse disabled (LANGFUSE_ENABLED=false)")
        _LANGFUSE_ACTIVE = False
        return False

    os.environ["LANGFUSE_PUBLIC_KEY"] = config.public_key
    os.environ["LANGFUSE_SECRET_KEY"] = config.secret_key
    os.environ["LANGFUSE_HOST"] = config.host

    _LANGFUSE_ACTIVE = True
    log.info("Langfuse enabled → %s", config.host)
    return True


def is_active() -> bool:
    return _LANGFUSE_ACTIVE


def get_observe() -> Callable:
    """Return the Langfuse @observe decorator, or a no-op passthrough."""
    if _LANGFUSE_ACTIVE:
        try:
            from langfuse import observe
            return observe
        except ImportError:
            log.warning("langfuse package not installed, tracing disabled")

    # No-op decorator
    def _noop_observe(**kwargs: Any) -> Callable:
        def decorator(fn: Callable) -> Callable:
            return fn
        return decorator
    return _noop_observe


def get_traced_azure_client_class():
    """Return the Langfuse-wrapped AzureOpenAI class, or the plain one."""
    if _LANGFUSE_ACTIVE:
        try:
            from langfuse.openai import AzureOpenAI
            log.debug("Using Langfuse-wrapped AzureOpenAI")
            return AzureOpenAI
        except ImportError:
            pass

    from openai import AzureOpenAI
    return AzureOpenAI


def get_traced_openai_client_class():
    """Return the Langfuse-wrapped OpenAI class, or the plain one."""
    if _LANGFUSE_ACTIVE:
        try:
            from langfuse.openai import OpenAI
            log.debug("Using Langfuse-wrapped OpenAI")
            return OpenAI
        except ImportError:
            pass

    from openai import OpenAI
    return OpenAI


def update_observation(**kwargs) -> None:
    """Attach metadata/output to the current Langfuse observation (no-op if inactive)."""
    if not _LANGFUSE_ACTIVE:
        return
    try:
        from langfuse import get_client
        ctx = get_client()
        ctx.update_current_observation(**kwargs)
    except Exception:
        log.debug("Langfuse update_observation failed (non-critical)", exc_info=True)


def flush() -> None:
    """Flush any pending Langfuse events."""
    if not _LANGFUSE_ACTIVE:
        return
    try:
        from langfuse import get_client
        client = get_client()
        client.flush()
    except Exception:
        log.debug("Langfuse flush failed (non-critical)", exc_info=True)
