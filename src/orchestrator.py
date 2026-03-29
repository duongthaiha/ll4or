"""Pipeline orchestrator — wires dataset → agents → execution → evaluation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from src.agents.debugger import DebuggerAgent
from src.agents.formulator import FormulatorAgent
from src.agents.heuristic_coder import HeuristicCoderAgent
from src.agents.hyperheuristic_coder import HyperHeuristicCoderAgent
from src.agents.metaheuristic_coder import MetaheuristicCoderAgent
from src.config import Config
from src.datasets.base import DatasetAdapter, Problem
from src.evaluation.evaluator import (
    AggregateMetrics,
    ComparisonResult,
    compare,
    compute_metrics,
)
from src.execution.sandbox import ExecutionResult, execute_code, extract_code
from src.llm.base import LLMClient
from src.tracing import get_observe

log = logging.getLogger(__name__)

_SOLVER_AGENTS = {
    "heuristic": HeuristicCoderAgent,
    "metaheuristic": MetaheuristicCoderAgent,
    "hyperheuristic": HyperHeuristicCoderAgent,
}


class Orchestrator:
    """End-to-end pipeline: dataset → formulate → solve → execute → evaluate."""

    def __init__(self, config: Config, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.formulator = FormulatorAgent(llm)
        self.debugger = DebuggerAgent(llm)
        self.solver_agents = {
            name: cls(llm)
            for name, cls in _SOLVER_AGENTS.items()
            if name in self.config.agent.solver_types
        }

        # Resolve the @observe decorator (no-op if Langfuse is off)
        self._observe = get_observe()

        # Wrap key methods with Langfuse tracing
        self.run = self._observe(name="pipeline_run")(self.run)
        self._solve_problem = self._observe(name="solve_problem")(self._solve_problem)
        self._run_solver = self._observe(name="run_solver")(self._run_solver)

    # ── public API ───────────────────────────────────────────────────

    def run(
        self,
        dataset: DatasetAdapter,
        max_problems: int | None = None,
    ) -> AggregateMetrics:
        """Run the full pipeline on a dataset and return aggregate metrics."""
        dataset.load()
        problems = dataset.get_problems()
        if max_problems:
            problems = problems[:max_problems]

        # Get dataset-specific evaluation config
        self._eval_config = dataset.get_eval_config()
        log.info(
            "Running %d problems from '%s' (eval: %s, tol=%s)",
            len(problems), dataset.name,
            self._eval_config.comparison_mode,
            self._eval_config.relative_tolerance
            if self._eval_config.comparison_mode != "absolute"
            else self._eval_config.absolute_tolerance,
        )

        all_results: list[dict] = []
        detail_records: list[dict] = []

        for i, problem in enumerate(problems):
            log.info(
                "[%d/%d] Problem %s", i + 1, len(problems), problem.id
            )
            results = self._solve_problem(problem)
            all_results.extend(results)
            detail_records.extend(results)

        metrics = compute_metrics(all_results)
        self._save_results(dataset.name, detail_records, metrics)
        self._print_summary(dataset.name, metrics)

        # Flush Langfuse traces
        from src.tracing import flush as langfuse_flush
        langfuse_flush()

        return metrics

    # ── internals ────────────────────────────────────────────────────

    def _solve_problem(self, problem: Problem) -> list[dict]:
        """Formulate → generate code (all solver types) → execute → evaluate."""

        # 1. Formulate
        input_data = {
            "problem_id": problem.id,
            "question": problem.question,
            "answer": problem.answer,
        }
        try:
            formulated = self.formulator.run(input_data)
        except Exception:
            log.exception("Formulation failed for problem %s", problem.id)
            formulated = {**input_data, "formulation": {}}

        # 2. Generate + execute for each solver type
        results: list[dict] = []
        for solver_name, solver_agent in self.solver_agents.items():
            result = self._run_solver(
                solver_name, solver_agent, formulated, problem
            )
            results.append(result)

        return results

    def _run_solver(
        self,
        solver_name: str,
        solver_agent,
        formulated: dict,
        problem: Problem,
    ) -> dict:
        """Generate code, execute, debug on failure, evaluate."""
        start = time.time()

        # Generate code
        try:
            generated = solver_agent.run(formulated)
            code = extract_code(generated.get("generated_code_raw", ""))
        except Exception:
            log.exception("%s code generation failed for %s", solver_name, problem.id)
            code = ""

        # Execute (with debug retries)
        exec_result: ExecutionResult | None = None
        for attempt in range(1 + self.config.agent.max_debug_retries):
            exec_result = execute_code(code, self.config.execution)
            if exec_result.success:
                break

            if attempt < self.config.agent.max_debug_retries and code:
                log.info(
                    "  %s attempt %d failed, trying debug fix…",
                    solver_name, attempt + 1,
                )
                error_msg = exec_result.stderr or exec_result.stdout or "No output"
                try:
                    fixed = self.debugger.run({
                        "question": problem.question,
                        "code": code,
                        "error": error_msg,
                    })
                    code = extract_code(fixed.get("fixed_code_raw", ""))
                except Exception:
                    log.exception("Debug attempt failed")
                    break

        elapsed = time.time() - start

        # Evaluate using dataset-specific eval config
        obj_val = exec_result.objective_value if exec_result else None
        raw_gt = problem.metadata.get("raw_answer")
        comp = compare(
            obj_val,
            problem.answer,
            self._eval_config,
            raw_ground_truth=raw_gt,
        )

        status_icon = "✓" if comp.is_correct else "✗"
        log.info(
            "  %s %s: predicted=%s, gt=%s (%s) [%.1fs]",
            status_icon, solver_name, obj_val, problem.answer,
            comp.detail, elapsed,
        )

        return {
            "problem_id": problem.id,
            "solver_type": solver_name,
            "execution_success": exec_result.success if exec_result else False,
            "objective_value": obj_val,
            "ground_truth": problem.answer,
            "comparison": comp,
            "elapsed_seconds": elapsed,
            "code": code,
            "stdout": exec_result.stdout if exec_result else "",
            "stderr": exec_result.stderr if exec_result else "",
        }

    # ── output ───────────────────────────────────────────────────────

    def _save_results(
        self,
        dataset_name: str,
        records: list[dict],
        metrics: AggregateMetrics,
    ) -> None:
        out_dir = self.config.output_dir / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Detailed results (JSONL)
        results_path = out_dir / "results.jsonl"
        with open(results_path, "w") as f:
            for r in records:
                row = {
                    "problem_id": r["problem_id"],
                    "solver_type": r["solver_type"],
                    "execution_success": r["execution_success"],
                    "objective_value": r["objective_value"],
                    "ground_truth": r["ground_truth"],
                    "is_correct": r["comparison"].is_correct,
                    "relative_error": r["comparison"].relative_error,
                    "detail": r["comparison"].detail,
                    "elapsed_seconds": r["elapsed_seconds"],
                }
                f.write(json.dumps(row) + "\n")

        # Metrics summary
        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

        log.info("Results saved to %s", out_dir)

    @staticmethod
    def _print_summary(dataset_name: str, metrics: AggregateMetrics) -> None:
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset_name}")
        print(f"  Total problems evaluated: {metrics.total}")
        print(f"  Overall accuracy: {metrics.accuracy:.1%}")
        print(f"  Failed executions: {metrics.failed_execution}")
        print(f"{'─'*60}")
        for solver, stats in metrics.per_solver.items():
            print(
                f"  {solver:20s}  "
                f"{stats['correct']}/{stats['total']} correct  "
                f"({stats['accuracy']:.1%})"
            )
        print(f"{'='*60}\n")
