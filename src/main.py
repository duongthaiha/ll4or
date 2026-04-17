"""CLI entry point for the multi-agent OR solver."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from src.config import Config
from src.datasets.registry import get_dataset, list_datasets
from src.llm.factory import create_llm_client
from src.orchestrator import Orchestrator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="or-solver",
        description="Multi-agent heuristic solver for Operations Research problems.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset to evaluate. Available: {list_datasets()}",
    )
    p.add_argument(
        "--solver",
        nargs="+",
        default=["heuristic", "metaheuristic", "hyperheuristic"],
        help="Solver types to run (default: all three).",
    )
    p.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Limit the number of problems to evaluate.",
    )
    p.add_argument(
        "--output",
        default="results",
        help="Output directory (default: results/).",
    )
    p.add_argument(
        "--llm-provider",
        default=None,
        help="LLM provider: openai, anthropic (overrides env).",
    )
    p.add_argument(
        "--llm-model",
        default=None,
        help="LLM model name (overrides env).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Code execution timeout in seconds (default: 600).",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Relative error tolerance (default: 0.05 = 5%%).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--sequential",
        action="store_true",
        help="Run solvers sequentially instead of in parallel.",
    )
    p.add_argument(
        "--parallel-problems",
        type=int,
        default=4,
        help="Number of problems to solve concurrently (default: 4).",
    )
    p.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit.",
    )

    # ── Multi-agent architecture flags ───────────────────────────────
    p.add_argument(
        "--no-analyze",
        action="store_true",
        help="Disable the Problem Analyzer agent (Phase 1).",
    )
    p.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Disable warm-start protocol (Phase 2: heuristic seeds meta/hyper).",
    )
    p.add_argument(
        "--no-critic",
        action="store_true",
        help="Disable the Code Critic agent (Phase 3: pre-execution review).",
    )
    p.add_argument(
        "--improve-iterations",
        type=int,
        default=2,
        help="Number of solution improvement iterations (0 to disable, default: 2).",
    )
    p.add_argument(
        "--no-selector",
        action="store_true",
        help="Disable the smart Ensemble Selector agent (Phase 5).",
    )
    p.add_argument(
        "--no-reflector",
        action="store_true",
        help="Disable the Reflector agent (Phase 6: cross-problem learning).",
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help="Run in legacy mode (disable all multi-agent enhancements).",
    )
    p.add_argument(
        "--install-solver-deps",
        action="store_true",
        help="Install common solver packages from requirements-solver.txt "
             "before running (pulls down numpy, scipy, pulp, ortools, deap, etc.).",
    )
    p.add_argument(
        "--skip-dep-check",
        action="store_true",
        help="Skip the startup check that warns about missing solver packages.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_datasets:
        print("Available datasets:")
        for name in list_datasets():
            print(f"  - {name}")
        return 0

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Initialize agent tracer (enabled if AGENT_TRACE_DIR env var is set)
    from src import agent_tracer
    agent_tracer.init_from_env()

    # Solver-package bootstrap: optionally install, otherwise just warn if missing.
    from src.execution.bootstrap import install_solver_deps, log_missing_solver_deps
    if args.install_solver_deps:
        install_solver_deps(missing_only=True)
    if not args.skip_dep_check:
        log_missing_solver_deps()

    # Build config
    config = Config.from_env()
    config.output_dir = __import__("pathlib").Path(args.output)
    config.execution.timeout = args.timeout
    config.evaluation.relative_tolerance = args.tolerance
    config.agent.solver_types = args.solver
    config.agent.parallel_problems = args.parallel_problems
    if args.sequential:
        config.agent.parallel_solvers = False

    # Multi-agent architecture settings
    if args.legacy:
        config.agent.enable_analyzer = False
        config.agent.enable_warm_start = False
        config.agent.enable_critic = False
        config.agent.improve_iterations = 0
        config.agent.enable_selector = False
        config.agent.enable_reflector = False
    else:
        if args.no_analyze:
            config.agent.enable_analyzer = False
        if args.no_warm_start:
            config.agent.enable_warm_start = False
        if args.no_critic:
            config.agent.enable_critic = False
        config.agent.improve_iterations = args.improve_iterations
        if args.no_selector:
            config.agent.enable_selector = False
        if args.no_reflector:
            config.agent.enable_reflector = False

    if args.llm_provider:
        config.llm.provider = args.llm_provider  # type: ignore[assignment]
        if args.llm_provider == "ollama":
            config.llm.base_url = os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434/v1"
            )
            config.llm.api_key = "ollama"
        elif args.llm_provider == "foundry":
            config.llm.base_url = os.environ.get("FOUNDRY_BASE_URL", "")
            config.llm.api_key = os.environ.get("FOUNDRY_API_KEY", "")
    if args.llm_model:
        config.llm.model = args.llm_model

    # Initialize Langfuse tracing (must happen before LLM client creation)
    from src.tracing import init_langfuse
    init_langfuse(config.langfuse)

    # Instantiate
    dataset = get_dataset(args.dataset)
    llm = create_llm_client(config.llm)
    orchestrator = Orchestrator(config, llm)

    # Run
    metrics = orchestrator.run(dataset, max_problems=args.max_problems)

    return 0 if metrics.accuracy > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
