"""Solution improver agent — iteratively refines the best solver solution.

Detects two distinct failure modes and applies different strategies:
  1. "All-same-wrong": all solvers agree on the same wrong answer → the
     mathematical formulation is likely wrong → re-read the problem from
     scratch and build a completely new solver (ignore previous code).
  2. "Close miss / divergent": solvers disagree or are close → refine the
     best solver code with parameter tuning and local search.
"""

from __future__ import annotations

import json
import logging

from src.agents.base import Agent

log = logging.getLogger(__name__)

# Used when solvers disagree or are close — refine existing approach
_SYSTEM_PROMPT_REFINE = """\
You are an expert optimization engineer. Multiple solvers attempted this \
problem but none produced a correct answer. You must write a BETTER solver.

Given:
1. The original problem description.
2. The mathematical formulation.
3. Results from previous solver attempts.
4. The code of the best solver so far.

Your task: write a new, improved Python solver script. Strategies:
- **Fix mathematical errors**: re-read the problem statement VERY carefully. \
  Check if constraints or objective were misinterpreted.
- **Increase iterations**: use up to 300 seconds of compute time (not 60).
- **Try a different algorithm**: if SA was used, try GA; if GA, try DE.
- **Better constraint handling**: replace penalty functions with repair operators.
- **Tighter convergence**: use smaller step sizes, more restarts.

RULES:
- Use ONLY Python standard library + numpy + scipy.
- The script must be **complete and runnable**.
- Print the result as: OBJECTIVE_VALUE: <number>
- Focus on CORRECTNESS over speed. You have up to 300 seconds.

Return ONLY the Python code inside a single ```python ... ``` block.
"""

# Used when ALL solvers agree on the same wrong answer — the formulation
# itself is almost certainly wrong. Ignore previous code entirely.
_SYSTEM_PROMPT_REFORMULATE = """\
You are an expert operations research analyst AND programmer.

Multiple independent solvers ALL produced the SAME WRONG answer for this \
problem. This strongly suggests the mathematical formulation or the problem \
interpretation is incorrect — NOT just a parameter tuning issue.

Your task: IGNORE the previous formulation and code entirely. Start fresh:
1. **Re-read the problem statement** word by word.
2. **Identify what was likely misunderstood**: look for tricky constraints, \
   hidden conditions, non-obvious variable domains, multi-objective aspects, \
   or data that was likely misread.
3. **Build a completely new mathematical model** from scratch.
4. **Write a new solver** based on your fresh formulation.

Common misunderstandings to watch for:
- Confusing "at most" (≤) with "at least" (≥)
- Missing a constraint entirely (e.g., capacity, budget, precedence)
- Wrong optimization direction (minimize vs maximize)
- Misreading numerical data (units, scaling factors, percentages vs fractions)
- Ignoring integer/binary requirements on decision variables
- Missing linked/coupled constraints between variables

RULES:
- Use ONLY Python standard library + numpy + scipy.
- The script must be **complete and runnable**.
- Print the result as: OBJECTIVE_VALUE: <number>
- Re-derive EVERYTHING from the problem text. Do NOT reuse previous code.
- Use up to 300 seconds of compute time.

Return ONLY the Python code inside a single ```python ... ``` block.
"""


def _detect_all_same_wrong(all_results: list[dict]) -> bool:
    """Check if all successful solvers produced the same wrong value."""
    successful = [
        r for r in all_results
        if r.get("execution_success") and r.get("objective_value") is not None
    ]
    if len(successful) < 2:
        return False

    values = [r["objective_value"] for r in successful]
    # Check if all values are identical (or within 0.1% of each other)
    ref = values[0]
    if ref == 0:
        return all(v == 0 for v in values)
    return all(abs(v - ref) / abs(ref) < 0.001 for v in values)


def _build_user_prompt_refine(
    question: str,
    formulation: dict,
    best_code: str,
    best_value: float | None,
    all_results: list[dict],
    iteration: int,
) -> str:
    form_str = json.dumps(formulation, indent=2, default=str)

    results_summary = []
    for r in all_results:
        status = "✓" if r.get("execution_success") else "✗"
        comp = r.get("comparison")
        correct = comp.is_correct if comp else False
        results_summary.append(
            f"  {status} {r.get('solver_type', '?')}: "
            f"obj={r.get('objective_value', 'N/A')}, "
            f"correct={correct}"
        )

    return (
        f"## Problem Description\n{question}\n\n"
        f"## Mathematical Formulation\n```json\n{form_str}\n```\n\n"
        f"## Previous Solver Results (attempt {iteration})\n"
        + "\n".join(results_summary) + "\n\n"
        f"## Best Objective Value So Far\n{best_value}\n\n"
        f"## Best Solver Code\n```python\n{best_code}\n```\n\n"
        "Write a better solver. Re-read the problem carefully — the answer "
        "is wrong, so something in the approach needs to change fundamentally."
    )


def _build_user_prompt_reformulate(
    question: str,
    wrong_value: float | None,
    all_results: list[dict],
    iteration: int,
) -> str:
    results_summary = []
    for r in all_results:
        results_summary.append(
            f"  {r.get('solver_type', '?')}: obj={r.get('objective_value', 'N/A')}"
        )

    return (
        f"## Problem Description (READ VERY CAREFULLY)\n{question}\n\n"
        f"## CRITICAL WARNING\n"
        f"All {len(all_results)} independent solvers produced the SAME wrong "
        f"answer: **{wrong_value}**. This means the problem was fundamentally "
        f"misunderstood. Do NOT trust ANY of the previous code or formulation.\n\n"
        f"## Previous Wrong Results\n" + "\n".join(results_summary) + "\n\n"
        f"## Attempt {iteration}\n"
        f"Start completely fresh. Re-read every word of the problem. "
        f"Build a new mathematical model and write a new solver from scratch."
    )


class ImproverAgent(Agent):
    """Generates an improved solver based on feedback from previous attempts.

    Detects "all-same-wrong" pattern (formulation error) vs "divergent/close"
    (algorithm error) and applies the appropriate strategy.
    """

    def run(self, input_data: dict) -> dict:
        question: str = input_data["question"]
        formulation: dict = input_data.get("formulation", {})
        best_code: str = input_data.get("best_code", "")
        best_value = input_data.get("best_value")
        all_results: list[dict] = input_data.get("all_results", [])
        iteration: int = input_data.get("iteration", 1)

        is_all_same = _detect_all_same_wrong(all_results)

        if is_all_same:
            log.info(
                "  Improver: all solvers agree on wrong value %s — "
                "triggering full re-formulation (attempt %d)",
                best_value, iteration,
            )
            system_prompt = _SYSTEM_PROMPT_REFORMULATE
            user_prompt = _build_user_prompt_reformulate(
                question, best_value, all_results, iteration
            )
        else:
            system_prompt = _SYSTEM_PROMPT_REFINE
            user_prompt = _build_user_prompt_refine(
                question, formulation, best_code, best_value,
                all_results, iteration,
            )

        raw = self._chat(system_prompt, user_prompt)

        return {
            **input_data,
            "solver_type": "improved",
            "generated_code_raw": raw,
            "improvement_mode": "reformulate" if is_all_same else "refine",
        }
