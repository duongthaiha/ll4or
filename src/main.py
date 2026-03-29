"""CLI entry point for the multi-agent OR solver."""

from __future__ import annotations

import argparse
import logging
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
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit.",
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

    # Build config
    config = Config.from_env()
    config.output_dir = __import__("pathlib").Path(args.output)
    config.execution.timeout = args.timeout
    config.evaluation.relative_tolerance = args.tolerance
    config.agent.solver_types = args.solver

    if args.llm_provider:
        config.llm.provider = args.llm_provider  # type: ignore[assignment]
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
