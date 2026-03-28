"""Evaluation — compare solver results against ground truth.

Supports multiple comparison modes matching each benchmark's reference eval:
  - "relative": ORLM-style (5% relative, optional integer rounding)
  - "absolute": BWOR/IndustryOR-style (absolute error < threshold)
  - "mamo_hybrid": MAMO-style (scale-based decimal check OR 0.01% relative)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import EvaluationConfig

log = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing a predicted value to ground truth."""

    predicted: float | None
    ground_truth: float | None
    is_correct: bool
    relative_error: float | None
    detail: str


def compare(
    predicted: float | None,
    ground_truth: float | None,
    config: EvaluationConfig | None = None,
    raw_predicted: str | None = None,
    raw_ground_truth: str | None = None,
) -> ComparisonResult:
    """Compare a solver's objective value against the expected answer.

    Args:
        predicted: Numeric result from code execution (None if failed).
        ground_truth: Expected numeric answer (None if unavailable).
        config: Dataset-specific evaluation settings.
        raw_predicted: Raw string output (for infeasible matching).
        raw_ground_truth: Raw string ground truth (for infeasible matching).
    """
    config = config or EvaluationConfig()

    # ── Handle infeasible / "No Best Solution" cases ─────────────
    gt_is_infeasible = (
        ground_truth is None
        and raw_ground_truth is not None
        and raw_ground_truth in config.infeasible_values
    )
    pred_is_infeasible = (
        predicted is None
        and raw_predicted is not None
        and raw_predicted in config.infeasible_values
    )

    if gt_is_infeasible:
        ok = pred_is_infeasible
        return ComparisonResult(
            predicted=predicted,
            ground_truth=ground_truth,
            is_correct=ok,
            relative_error=None,
            detail=f"Infeasible: gt='{raw_ground_truth}', pred='{raw_predicted}', match={ok}",
        )

    # ── No prediction ────────────────────────────────────────────
    if predicted is None:
        return ComparisonResult(
            predicted=predicted,
            ground_truth=ground_truth,
            is_correct=False,
            relative_error=None,
            detail="No predicted value",
        )

    if ground_truth is None:
        return ComparisonResult(
            predicted=predicted,
            ground_truth=ground_truth,
            is_correct=False,
            relative_error=None,
            detail="No ground truth available",
        )

    # ── Dispatch to comparison mode ──────────────────────────────
    mode = config.comparison_mode
    if mode == "relative":
        return _compare_relative(predicted, ground_truth, config)
    elif mode == "absolute":
        return _compare_absolute(predicted, ground_truth, config)
    elif mode == "mamo_hybrid":
        return _compare_mamo_hybrid(predicted, ground_truth, config, raw_ground_truth)
    else:
        raise ValueError(f"Unknown comparison_mode: {mode}")


# ── Comparison mode implementations ─────────────────────────────────


def _compare_relative(
    predicted: float, ground_truth: float, config: EvaluationConfig
) -> ComparisonResult:
    """ORLM-style: optional integer rounding, then relative error <= tolerance."""
    p, gt = predicted, ground_truth

    if config.round_to_int:
        p = round(p)
        gt = round(gt)

    if gt == 0:
        abs_err = abs(p)
        ok = abs_err <= config.absolute_tolerance
        return ComparisonResult(
            predicted=predicted,
            ground_truth=ground_truth,
            is_correct=ok,
            relative_error=None,
            detail=f"GT=0, abs_error={abs_err:.6f}, tol={config.absolute_tolerance} (rounded={config.round_to_int})",
        )

    rel_err = abs((p - gt) / gt)
    ok = rel_err <= config.relative_tolerance
    return ComparisonResult(
        predicted=predicted,
        ground_truth=ground_truth,
        is_correct=ok,
        relative_error=rel_err,
        detail=f"rel_error={rel_err:.4%}, tol={config.relative_tolerance:.2%} (rounded={config.round_to_int})",
    )


def _compare_absolute(
    predicted: float, ground_truth: float, config: EvaluationConfig
) -> ComparisonResult:
    """BWOR/IndustryOR-style: absolute error < tolerance."""
    abs_err = abs(predicted - ground_truth)
    ok = abs_err < config.absolute_tolerance
    rel_err = abs((predicted - ground_truth) / ground_truth) if ground_truth != 0 else None
    return ComparisonResult(
        predicted=predicted,
        ground_truth=ground_truth,
        is_correct=ok,
        relative_error=rel_err,
        detail=f"abs_error={abs_err:.6f}, tol={config.absolute_tolerance}",
    )


def _compare_mamo_hybrid(
    predicted: float,
    ground_truth: float,
    config: EvaluationConfig,
    raw_ground_truth: str | None = None,
) -> ComparisonResult:
    """MAMO-style: scale-based decimal check OR relative error <= 1e-4.

    Reference: MAMO's comp() + compare_output_with_standard()
    1. Determine decimal digits from the raw GT string.
    2. Multiply both values by 10^digits, check if |diff| < 1.
    3. OR check relative error <= 1e-4.
    Either passing means correct.
    """
    # Determine decimal digits from raw string
    gt_str = raw_ground_truth or str(ground_truth)
    if "." in gt_str:
        decimal_digits = len(gt_str.rstrip("0").split(".")[-1])
        decimal_digits = max(decimal_digits, 2)
    else:
        decimal_digits = 2

    scale = 10 ** decimal_digits
    scaled_diff = abs(predicted * scale - ground_truth * scale)
    scale_ok = scaled_diff < 1

    # comp() from MAMO reference
    diff = abs(predicted - ground_truth)
    if ground_truth == 0:
        rate = diff * 1e-4
    else:
        rate = diff / abs(ground_truth)
    rel_ok = abs(rate) <= 1e-4

    ok = scale_ok or rel_ok
    rel_err = rate if ground_truth != 0 else None

    return ComparisonResult(
        predicted=predicted,
        ground_truth=ground_truth,
        is_correct=ok,
        relative_error=rel_err,
        detail=f"mamo: scale_diff={scaled_diff:.4f}(<1={scale_ok}), rate={rate:.6f}(<=1e-4={rel_ok})",
    )


# ── Aggregate metrics ────────────────────────────────────────────────


@dataclass
class AggregateMetrics:
    """Summary metrics across a set of problems."""

    total: int = 0
    correct: int = 0
    failed_execution: int = 0
    no_ground_truth: int = 0
    accuracy: float = 0.0
    per_solver: dict[str, dict] = field(default_factory=dict)


def compute_metrics(results: list[dict]) -> AggregateMetrics:
    """Compute aggregate metrics from a list of solver result dicts.

    Each dict is expected to have:
      - solver_type: str
      - comparison: ComparisonResult
      - execution_success: bool
    """
    m = AggregateMetrics(total=len(results))
    solver_buckets: dict[str, list[bool]] = {}

    for r in results:
        comp: ComparisonResult = r["comparison"]
        solver = r.get("solver_type", "unknown")

        if comp.ground_truth is None:
            m.no_ground_truth += 1
            continue

        if not r.get("execution_success", False):
            m.failed_execution += 1

        if comp.is_correct:
            m.correct += 1

        solver_buckets.setdefault(solver, []).append(comp.is_correct)

    m.accuracy = m.correct / max(m.total - m.no_ground_truth, 1)

    for solver, outcomes in solver_buckets.items():
        n = len(outcomes)
        c = sum(outcomes)
        m.per_solver[solver] = {
            "total": n,
            "correct": c,
            "accuracy": c / max(n, 1),
        }

    return m
