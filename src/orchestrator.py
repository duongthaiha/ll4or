"""Pipeline orchestrator — wires dataset → agents → execution → evaluation.

Supports the full multi-agent architecture:
  1. Problem Analyzer (classify, route)
  2. Formulator (NL → math)
  3. Solver agents (heuristic first for warm-start, then meta/hyper in parallel)
  4. Code Critic (pre-execution review)
  5. Execution + debug retries
  6. Solution Improver (iterative refinement)
  7. Ensemble Selector (smart pick)
  8. Reflector (cross-problem learning)

Each phase can be toggled via AgentConfig flags.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from src import agent_tracer
from src.agents.analyzer import AnalyzerAgent
from src.agents.critic import CriticAgent
from src.agents.debugger import DebuggerAgent
from src.agents.formulator import FormulatorAgent
from src.agents.heuristic_coder import HeuristicCoderAgent
from src.agents.hyperheuristic_coder import HyperHeuristicCoderAgent
from src.agents.improver import ImproverAgent
from src.agents.metaheuristic_coder import MetaheuristicCoderAgent
from src.agents.reflector import ReflectorAgent
from src.agents.researcher import ResearcherAgent
from src.agents.selector import SelectorAgent
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
from src.tracing import get_observe, update_observation

log = logging.getLogger(__name__)

_SOLVER_AGENTS = {
    "heuristic": HeuristicCoderAgent,
    "metaheuristic": MetaheuristicCoderAgent,
    "hyperheuristic": HyperHeuristicCoderAgent,
}


class Orchestrator:
    """End-to-end pipeline: dataset → formulate → solve → execute → evaluate.

    Supports the hierarchical multi-agent architecture with adaptive routing,
    warm-start, code critic, solution improvement, smart ensemble selection,
    and cross-problem learning.
    """

    def __init__(self, config: Config, llm: LLMClient):
        self.config = config
        self.llm = llm

        # Core agents (always present)
        self.formulator = FormulatorAgent(llm)
        self.debugger = DebuggerAgent(llm)
        self.solver_agents = {
            name: cls(llm)
            for name, cls in _SOLVER_AGENTS.items()
            if name in self.config.agent.solver_types
        }

        # Multi-agent architecture agents (conditionally created)
        self.analyzer = AnalyzerAgent(llm) if config.agent.enable_analyzer else None
        self.researcher = (
            ResearcherAgent(llm, kb_path=config.agent.researcher_kb_path)
            if config.agent.enable_researcher
            else None
        )
        self.critic = CriticAgent(llm) if config.agent.enable_critic else None
        self.improver = ImproverAgent(llm) if config.agent.improve_iterations > 0 else None
        self.selector = SelectorAgent(llm) if config.agent.enable_selector else None
        self.reflector = ReflectorAgent(llm) if config.agent.enable_reflector else None

        # Cross-problem learning state
        self._accumulated_lessons: list[dict] = []

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

        # Log active phases
        phases = []
        if self.analyzer:
            phases.append("analyzer")
        if self.researcher:
            phases.append("researcher")
        if self.config.agent.enable_warm_start:
            phases.append("warm-start")
        if self.critic:
            phases.append("critic")
        if self.improver:
            phases.append(f"improver(×{self.config.agent.improve_iterations})")
        if self.selector:
            phases.append("selector")
        if self.reflector:
            phases.append("reflector")

        log.info(
            "Running %d problems from '%s' (eval: %s, tol=%s, phases: %s)",
            len(problems), dataset.name,
            self._eval_config.comparison_mode,
            self._eval_config.relative_tolerance
            if self._eval_config.comparison_mode != "absolute"
            else self._eval_config.absolute_tolerance,
            ", ".join(phases) if phases else "legacy",
        )

        all_results: list[dict] = []

        n_parallel = self.config.agent.parallel_problems
        if n_parallel > 1 and len(problems) > 1:
            log.info("Running %d problems concurrently", n_parallel)
            all_results = self._run_problems_parallel(problems, n_parallel)
        else:
            for i, problem in enumerate(problems):
                log.info(
                    "[%d/%d] Problem %s", i + 1, len(problems), problem.id
                )
                results = self._solve_problem(problem)
                all_results.extend(results)

        metrics = compute_metrics(all_results)
        self._save_results(dataset.name, all_results, metrics)
        self._print_summary(dataset.name, metrics)

        # Flush Langfuse traces
        from src.tracing import flush as langfuse_flush
        langfuse_flush()

        return metrics

    # ── internals ────────────────────────────────────────────────────

    def _run_problems_parallel(
        self, problems: list[Problem], max_workers: int
    ) -> list[dict]:
        """Run multiple problems concurrently."""
        import threading

        all_results: list[dict] = []
        lock = threading.Lock()
        done_count = [0]
        total = len(problems)

        def solve_one(problem: Problem, **kwargs) -> list[dict]:
            results = self._solve_problem(problem, **kwargs)
            with lock:
                all_results.extend(results)
                done_count[0] += 1
                log.info(
                    "  [%d/%d completed] Problem %s",
                    done_count[0], total, problem.id,
                )
            return results

        # Capture Langfuse context for propagation into problem threads
        langfuse_ctx = {}
        try:
            from langfuse import get_client
            lf = get_client()
            trace_id = lf.get_current_trace_id()
            obs_id = lf.get_current_observation_id()
            if trace_id:
                langfuse_ctx["langfuse_trace_id"] = trace_id
            if obs_id:
                langfuse_ctx["langfuse_parent_observation_id"] = obs_id
        except Exception:
            pass

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(solve_one, p, **langfuse_ctx)
                for p in problems
            ]
            # Wait for all to complete; exceptions are re-raised
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    log.exception("Problem solver failed")

        return all_results

    def _solve_problem(self, problem: Problem, **kwargs) -> list[dict]:
        """Full multi-agent pipeline for a single problem.

        Phases:
          1. Analyze (if enabled) — classify problem type and recommend strategy
          2. Formulate — NL → structured math formulation
          3. Solve — generate and execute code (with warm-start if enabled)
          4. Improve — iteratively refine best solution (if enabled)
          5. Reflect — extract lessons for future problems (if enabled)
        """
        input_data = {
            "problem_id": problem.id,
            "question": problem.question,
            "answer": problem.answer,
        }
        agent_tracer.set_problem(str(problem.id))

        # ── Phase 1: Analyze ─────────────────────────────────────────
        analysis = {}
        if self.analyzer:
            try:
                analyzed = self.analyzer.run(input_data)
                analysis = analyzed.get("analysis", {})
                log.info(
                    "  Analyzer: class=%s, difficulty=%s, recommended_meta=%s",
                    analysis.get("problem_class", "?"),
                    analysis.get("difficulty", "?"),
                    analysis.get("recommended_solvers", {}).get(
                        "metaheuristic_algorithm", "?"
                    ),
                )
            except Exception:
                log.exception("Analysis failed for problem %s", problem.id)

        # ── Phase 1b (v4): Research ──────────────────────────────────
        research: dict = {}
        if self.researcher:
            try:
                researched = self.researcher.run({
                    **input_data,
                    "analysis": analysis,
                })
                research = researched.get("research", {}) or {}
                log.info(
                    "  Researcher: canonical=%s, heuristic=%s%s",
                    research.get("canonical_name", "?"),
                    (research.get("heuristic_methods") or ["?"])[0],
                    f" [kb:{research.get('_kb_match')}]" if research.get("_kb_match") else "",
                )
            except Exception:
                log.exception("Research failed for problem %s", problem.id)

        # ── Phase 2: Formulate ───────────────────────────────────────
        try:
            formulated = self.formulator.run(input_data)
        except Exception:
            log.exception("Formulation failed for problem %s", problem.id)
            formulated = {**input_data, "formulation": {}}

        # Inject analysis + research into formulated data for downstream agents
        formulated["analysis"] = analysis
        formulated["research"] = research

        # ── Phase 3: Solve ───────────────────────────────────────────
        if self.config.agent.enable_warm_start and "heuristic" in self.solver_agents:
            results = self._run_solvers_warm_start(formulated, problem)
        elif self.config.agent.parallel_solvers and len(self.solver_agents) > 1:
            results = self._run_solvers_parallel(formulated, problem)
        else:
            results = [
                self._run_solver(name, agent, formulated, problem)
                for name, agent in self.solver_agents.items()
            ]

        # ── Phase 4: Improve ─────────────────────────────────────────
        if self.improver and self.config.agent.improve_iterations > 0:
            results = self._run_improvement_loop(
                formulated, problem, results
            )

        # ── Phase 5: Reflect ─────────────────────────────────────────
        any_correct = any(
            r.get("comparison", ComparisonResult(None, None, False, None, "")).is_correct
            for r in results
        )
        if self.reflector:
            try:
                reflection_result = self.reflector.run({
                    "question": problem.question,
                    "analysis": analysis,
                    "results": [
                        {
                            "solver_type": r.get("solver_type"),
                            "objective_value": r.get("objective_value"),
                            "execution_success": r.get("execution_success"),
                            "comparison": r.get("comparison"),
                        }
                        for r in results
                    ],
                    "is_correct": any_correct,
                    "prior_lessons": self._accumulated_lessons,
                })
                reflection = reflection_result.get("reflection", {})
                new_lessons = reflection.get("lessons", [])
                if new_lessons:
                    self._accumulated_lessons.extend(new_lessons)
                    log.info(
                        "  Reflector: %d new lessons (total: %d)",
                        len(new_lessons), len(self._accumulated_lessons),
                    )
            except Exception:
                log.exception("Reflection failed for problem %s", problem.id)

        return results

    def _run_solvers_warm_start(
        self, formulated: dict, problem: Problem
    ) -> list[dict]:
        """Run heuristic FIRST, then use its result to warm-start meta/hyper.

        Phase 1: Heuristic runs alone (fast, constructive)
        Phase 2: Meta + hyper run in parallel, seeded with heuristic result
        """
        results: list[dict] = []

        # Phase 1: Run heuristic first
        heuristic_result = self._run_solver(
            "heuristic", self.solver_agents["heuristic"], formulated, problem
        )
        results.append(heuristic_result)

        # Build warm-start context from heuristic result
        warm_start = None
        if heuristic_result.get("execution_success") and heuristic_result.get("objective_value") is not None:
            warm_start = {
                "objective_value": heuristic_result["objective_value"],
                "solver_type": "heuristic",
            }
            log.info(
                "  Warm-start: heuristic found obj=%s, seeding meta/hyper",
                warm_start["objective_value"],
            )

        # Phase 2: Run remaining solvers with warm-start (parallel if enabled)
        remaining_agents = {
            name: agent
            for name, agent in self.solver_agents.items()
            if name != "heuristic"
        }

        if not remaining_agents:
            return results

        # Inject warm-start into formulated data
        formulated_with_ws = {**formulated, "warm_start": warm_start}

        if self.config.agent.parallel_solvers and len(remaining_agents) > 1:
            # Run remaining solvers in parallel with warm-start
            langfuse_ctx = {}
            try:
                from langfuse import get_client
                lf = get_client()
                trace_id = lf.get_current_trace_id()
                obs_id = lf.get_current_observation_id()
                if trace_id:
                    langfuse_ctx["langfuse_trace_id"] = trace_id
                if obs_id:
                    langfuse_ctx["langfuse_parent_observation_id"] = obs_id
            except Exception:
                pass

            with ThreadPoolExecutor(max_workers=len(remaining_agents)) as pool:
                futures = {
                    pool.submit(
                        self._run_solver, name, agent, formulated_with_ws, problem,
                        **langfuse_ctx,
                    ): name
                    for name, agent in remaining_agents.items()
                }
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            for name, agent in remaining_agents.items():
                results.append(
                    self._run_solver(name, agent, formulated_with_ws, problem)
                )

        return results

    def _run_solvers_parallel(
        self, formulated: dict, problem: Problem
    ) -> list[dict]:
        """Run all solver agents concurrently using threads.

        Captures the current Langfuse trace/observation IDs and passes them
        to each thread so child spans remain grouped under the parent trace.
        """
        # Capture Langfuse context from the current thread for propagation
        langfuse_ctx = {}
        try:
            from langfuse import get_client
            lf = get_client()
            trace_id = lf.get_current_trace_id()
            obs_id = lf.get_current_observation_id()
            if trace_id:
                langfuse_ctx["langfuse_trace_id"] = trace_id
            if obs_id:
                langfuse_ctx["langfuse_parent_observation_id"] = obs_id
        except Exception:
            pass  # Langfuse not active or not available

        results: list[dict] = []
        with ThreadPoolExecutor(
            max_workers=len(self.solver_agents)
        ) as pool:
            futures = {
                pool.submit(
                    self._run_solver, name, agent, formulated, problem,
                    **langfuse_ctx,
                ): name
                for name, agent in self.solver_agents.items()
            }
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def _run_solver(
        self,
        solver_name: str,
        solver_agent,
        formulated: dict,
        problem: Problem,
        **kwargs,
    ) -> dict:
        """Generate code, optionally critique, execute, debug on failure, evaluate."""
        start = time.time()

        # Generate code
        try:
            generated = solver_agent.run(formulated)
            code = extract_code(generated.get("generated_code_raw", ""))
        except Exception:
            log.exception("%s code generation failed for %s", solver_name, problem.id)
            code = ""

        # ── Critic review (Phase 3) ──────────────────────────────────
        if self.critic and code:
            try:
                review_result = self.critic.run({
                    "question": problem.question,
                    "formulation": formulated.get("formulation", {}),
                    "code": code,
                    "solver_type": solver_name,
                })
                review = review_result.get("review", {})
                critical_issues = [
                    i for i in review.get("issues", [])
                    if i.get("severity") == "critical"
                ]
                if critical_issues and not review.get("approved", True):
                    log.info(
                        "  Critic found %d critical issues in %s, requesting fix",
                        len(critical_issues), solver_name,
                    )
                    # Use debugger to fix critic-identified issues
                    issue_desc = "\n".join(
                        f"- [{i['category']}] {i['description']} (fix: {i.get('fix', 'N/A')})"
                        for i in critical_issues
                    )
                    try:
                        fixed = self.debugger.run({
                            "question": problem.question,
                            "code": code,
                            "error": f"Code review found these issues:\n{issue_desc}",
                            "research": formulated.get("research", {}),
                        })
                        code = extract_code(fixed.get("fixed_code_raw", ""))
                    except Exception:
                        log.exception("Critic-triggered fix failed")
            except Exception:
                log.exception("Critic review failed for %s", solver_name)

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
                        "research": formulated.get("research", {}),
                    })
                    code = extract_code(fixed.get("fixed_code_raw", ""))
                except Exception:
                    log.exception("Debug attempt failed")
                    break

        elapsed = time.time() - start

        # Log code & execution result to Langfuse
        update_observation(metadata={
            "generated_code": code,
            "execution": {
                "success": exec_result.success if exec_result else False,
                "objective_value": exec_result.objective_value if exec_result else None,
                "stdout": exec_result.stdout if exec_result else "",
                "stderr": exec_result.stderr if exec_result else "",
                "return_code": exec_result.return_code if exec_result else -1,
                "timed_out": exec_result.timed_out if exec_result else False,
            },
        })

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
            "research": formulated.get("research", {}),
        }

    def _run_improvement_loop(
        self,
        formulated: dict,
        problem: Problem,
        results: list[dict],
    ) -> list[dict]:
        """Run iterative improvement on the best solver result (Phase 4).

        Detects failure mode:
          - "all-same-wrong": all solvers agree on wrong value → re-formulate
          - "close/divergent": solvers disagree → refine best code

        Each iteration generates new code, applies debug retries if needed,
        and evaluates against ground truth.
        """
        # Find the best result (prefer correct answers, then best objective)
        successful = [r for r in results if r.get("execution_success")]
        if not successful:
            log.info("  Improver: skipping — no successful executions to improve")
            return results

        # Sort by correctness (correct first), then by objective closeness to GT
        def _sort_key(r: dict) -> tuple:
            comp = r.get("comparison", ComparisonResult(None, None, False, None, ""))
            return (
                not comp.is_correct,  # correct first (False < True)
                abs(comp.relative_error) if comp.relative_error is not None else float("inf"),
            )

        best = min(successful, key=_sort_key)

        # Skip improvement if best is already correct
        if best.get("comparison", ComparisonResult(None, None, False, None, "")).is_correct:
            log.info("  Improver: skipping — already have correct answer")
            return results

        consecutive_failures = 0
        max_consecutive_failures = 3

        for iteration in range(self.config.agent.improve_iterations):
            log.info(
                "  Improver iteration %d/%d (best so far: %s from %s)",
                iteration + 1,
                self.config.agent.improve_iterations,
                best.get("objective_value"),
                best.get("solver_type"),
            )

            try:
                improved = self.improver.run({
                    "question": problem.question,
                    "formulation": formulated.get("formulation", {}),
                    "best_code": best.get("code", ""),
                    "best_value": best.get("objective_value"),
                    "all_results": [
                        {
                            "solver_type": r.get("solver_type"),
                            "objective_value": r.get("objective_value"),
                            "execution_success": r.get("execution_success"),
                            "comparison": r.get("comparison"),
                        }
                        for r in results
                    ],
                    "iteration": iteration + 1,
                })
                improved_code = extract_code(improved.get("generated_code_raw", ""))
                improvement_mode = improved.get("improvement_mode", "refine")
            except Exception:
                log.exception("Improver iteration %d failed", iteration + 1)
                continue  # try next iteration instead of breaking

            if not improved_code:
                log.info("  Improver iteration %d: no code generated", iteration + 1)
                continue

            # Execute with debug retries (improved code may also have bugs)
            exec_result: ExecutionResult | None = None
            code = improved_code
            for attempt in range(1 + min(self.config.agent.max_debug_retries, 2)):
                exec_result = execute_code(code, self.config.execution)
                if exec_result.success:
                    break
                if attempt < min(self.config.agent.max_debug_retries, 2) and code:
                    error_msg = exec_result.stderr or exec_result.stdout or "No output"
                    try:
                        fixed = self.debugger.run({
                            "question": problem.question,
                            "code": code,
                            "error": error_msg,
                            "research": formulated.get("research", {}),
                        })
                        code = extract_code(fixed.get("fixed_code_raw", ""))
                    except Exception:
                        break

            if not exec_result or not exec_result.success:
                consecutive_failures += 1
                log.info(
                    "  Improver iteration %d (%s): execution failed (%d consecutive)",
                    iteration + 1, improvement_mode, consecutive_failures,
                )
                if consecutive_failures >= max_consecutive_failures:
                    log.info(
                        "  Improver: bailing out after %d consecutive execution failures",
                        consecutive_failures,
                    )
                    break
                continue

            consecutive_failures = 0  # reset on success

            # Evaluate
            raw_gt = problem.metadata.get("raw_answer")
            comp = compare(
                exec_result.objective_value,
                problem.answer,
                self._eval_config,
                raw_ground_truth=raw_gt,
            )

            improved_result = {
                "problem_id": problem.id,
                "solver_type": f"improved_v{iteration + 1}",
                "execution_success": True,
                "objective_value": exec_result.objective_value,
                "ground_truth": problem.answer,
                "comparison": comp,
                "elapsed_seconds": 0.0,
                "code": code,
                "stdout": exec_result.stdout,
                "stderr": exec_result.stderr,
            }
            results.append(improved_result)

            status_icon = "✓" if comp.is_correct else "✗"
            log.info(
                "  %s improved_v%d (%s): predicted=%s, gt=%s (%s)",
                status_icon, iteration + 1, improvement_mode,
                exec_result.objective_value, problem.answer, comp.detail,
            )

            # Update best if improved
            if comp.is_correct:
                log.info("  Improver: found correct answer on iteration %d!", iteration + 1)
                break
            elif _sort_key(improved_result) < _sort_key(best):
                best = improved_result

        return results

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
                    "code": r["code"],
                    "research": r.get("research", {}),
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
