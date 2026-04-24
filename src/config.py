"""Configuration module for the multi-agent OR solver."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: Literal["openai", "anthropic", "azure", "ollama", "foundry"] = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str | None = None
    api_version: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096

    def __post_init__(self):
        if not self.api_key:
            if self.provider == "ollama":
                self.api_key = "ollama"  # Ollama needs no real key
            else:
                env_map = {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "azure": "AZURE_OPENAI_API_KEY",
                }
                env_var = env_map.get(self.provider, "LLM_API_KEY")
                self.api_key = os.environ.get(env_var, "")


@dataclass
class ExecutionConfig:
    """Code execution settings."""

    timeout: int = 600
    max_workers: int = 8
    allowed_imports: list[str] = field(
        default_factory=lambda: [
            "numpy", "scipy", "math", "random", "itertools",
            "functools", "collections", "heapq", "copy", "typing",
            "dataclasses", "abc", "operator", "sys", "time",
        ]
    )


@dataclass
class EvaluationConfig:
    """Evaluation and comparison settings.

    comparison_mode controls how predicted vs ground truth are compared:
      - "relative": |pred - gt| / |gt| <= relative_tolerance  (ORLM default)
      - "absolute": |pred - gt| < absolute_tolerance           (BWOR / IndustryOR)
      - "mamo_hybrid": scale-based decimal check OR relative <= 1e-4  (MAMO)
    """

    comparison_mode: Literal["relative", "absolute", "mamo_hybrid"] = "relative"
    relative_tolerance: float = 0.05
    absolute_tolerance: float = 0.05
    round_to_int: bool = False
    infeasible_values: list[str] = field(
        default_factory=lambda: ["No Best Solution"]
    )
    pass_k_values: list[int] = field(default_factory=lambda: [1])


@dataclass
class AgentConfig:
    """Agent behavior settings."""

    max_debug_retries: int = 3
    solver_types: list[str] = field(
        default_factory=lambda: ["heuristic", "metaheuristic", "hyperheuristic"]
    )
    parallel_solvers: bool = True
    parallel_problems: int = 1  # number of problems to solve concurrently

    # ── Multi-agent architecture settings ────────────────────────────
    enable_analyzer: bool = True       # Phase 1: classify problem before solving
    enable_researcher: bool = True     # Phase 1b (v4): literature-grounded dossier
    researcher_kb_path: str | None = None  # override for curated KB JSON file
    enable_warm_start: bool = True     # Phase 2: heuristic → meta/hyper warm-start
    enable_critic: bool = True         # Phase 3: review code before execution
    improve_iterations: int = 2        # Phase 4: LLM-guided improvement iterations (0=off)
    enable_selector: bool = True       # Phase 5: smart ensemble selection
    enable_reflector: bool = True      # Phase 6: cross-problem learning


@dataclass
class LangfuseConfig:
    """Langfuse observability settings."""

    enabled: bool = False
    host: str = "http://localhost:3000"
    public_key: str = ""
    secret_key: str = ""


@dataclass
class Config:
    """Top-level configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)
    output_dir: Path = Path("results")
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Config:
        """Build config from environment variables (loads .env if present)."""
        # Load .env file if it exists
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            _load_dotenv(env_path)

        provider = os.environ.get("LLM_PROVIDER", "openai")
        llm = LLMConfig(
            provider=provider,  # type: ignore[arg-type]
            model=os.environ.get("LLM_MODEL", "gpt-4o"),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
        )
        # Azure-specific settings
        if provider == "azure":
            llm.base_url = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            llm.api_version = os.environ.get("AZURE_API_VERSION", "2025-04-01-preview")
            llm.api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        # Ollama-specific settings
        elif provider == "ollama":
            llm.base_url = os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434/v1"
            )
            llm.api_key = "ollama"
        # Azure AI Foundry (Models-as-a-Service) settings
        elif provider == "foundry":
            llm.base_url = os.environ.get("FOUNDRY_BASE_URL", "")
            llm.api_key = os.environ.get("FOUNDRY_API_KEY", "")

        execution = ExecutionConfig(
            timeout=int(os.environ.get("EXEC_TIMEOUT", "600")),
            max_workers=int(os.environ.get("EXEC_MAX_WORKERS", "8")),
        )
        evaluation = EvaluationConfig(
            relative_tolerance=float(os.environ.get("EVAL_REL_TOL", "0.05")),
        )
        langfuse = LangfuseConfig(
            enabled=os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true",
            host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        )
        return cls(
            llm=llm,
            execution=execution,
            evaluation=evaluation,
            langfuse=langfuse,
            output_dir=Path(os.environ.get("OUTPUT_DIR", "results")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — sets vars that aren't already in the environment."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value
