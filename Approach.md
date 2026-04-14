# V3 Multi-Agent Architecture — Detailed Approach

## Overview

ll4or uses a **hierarchical multi-agent pipeline** to solve Operations Research problems. Instead of asking a single LLM to solve a problem in one shot, the system decomposes the task across **10 specialized agents**, each handling a distinct phase of the solution process. This decomposition is critical: it allows each agent to focus on what it does best, enables error recovery at every stage, and produces multiple independent solutions that can be compared and improved.

The pipeline processes each problem through six sequential phases:

```
Phase 1: ANALYZE → Phase 2: FORMULATE → Phase 3: SOLVE → Phase 4: IMPROVE → Phase 5: REFLECT
```

Within Phase 3, three solver agents work in parallel (or via warm-start sequencing), each generating a different algorithmic approach. The result is an ensemble of solutions that dramatically outperforms any single solver.

---

## Why Multiple Agents?

A single LLM call to "solve this optimization problem" fails frequently because it must simultaneously:
1. Understand the natural language problem
2. Identify the mathematical structure
3. Choose an appropriate algorithm
4. Implement correct Python code
5. Handle edge cases and constraints

By splitting these responsibilities across specialized agents, each step can be independently verified, debugged, and improved. Key benefits:

- **Error isolation:** A formulation error doesn't contaminate algorithm selection
- **Diversity:** Three independent solvers approach the problem differently, increasing the chance that at least one succeeds
- **Recovery:** The Critic catches bugs before execution; the Debugger fixes failures after; the Improver rethinks the approach entirely when all solvers fail
- **Learning:** The Reflector accumulates lessons across problems, improving later solutions

---

## The Agents

### 1. Analyzer Agent
**File:** `src/agents/analyzer.py`  
**Role:** Problem classification and strategy recommendation  
**Temperature:** 0.2 (deterministic)

The Analyzer is the first agent to see each problem. It classifies the problem type (LP, IP, MIP, NLP, combinatorial, scheduling, knapsack, etc.), estimates difficulty (easy/medium/hard), and recommends specific algorithms for downstream solvers.

**Input:** Natural-language problem description  
**Output:** Structured analysis including:
- `problem_class` — LP, IP, MIP, NLP, combinatorial, network_flow, scheduling, assignment, knapsack, TSP, VRP
- `difficulty` — easy, medium, hard
- `scale` — estimated number of variables/constraints, whether it's large-scale
- `structure` — linearity, integer/binary variables, convexity
- `recommended_solvers` — specific heuristic strategy, metaheuristic algorithm (GA, SA, PSO, Tabu Search, DE, ALNS, ACO), and suggested parameters (population size, iterations, cooling rate)
- `key_challenges` — potential difficulties for solvers

**Why it matters:** Without the Analyzer, all three solvers would guess blindly at what algorithm to use. The Analyzer's recommendations guide solver agents toward appropriate approaches — e.g., suggesting Simulated Annealing for continuous NLP problems or Genetic Algorithms for combinatorial problems.

### 2. Formulator Agent
**File:** `src/agents/formulator.py`  
**Role:** Translate natural language into mathematical formulation  
**Temperature:** 0.2 (deterministic)

The Formulator converts the problem description into a structured mathematical model. This is arguably the most critical step — if the formulation is wrong, all downstream code will solve the wrong problem.

**Input:** Problem description  
**Output:** Structured JSON formulation:
- `problem_type` — LP, IP, MIP, NLP, combinatorial
- `objective` — direction (minimize/maximize) and description
- `decision_variables` — name, type (continuous/integer/binary), meaning
- `constraints` — plain-English descriptions
- `parameters` — names and values from the problem
- `summary` — one-sentence problem summary

**Why it matters:** The formulation serves as a shared contract between all downstream agents. Every solver reads the same structured formulation, ensuring they're all attempting to solve the same mathematical problem. When the formulation is wrong, the Improver can detect this (via the "all-same-wrong" pattern) and trigger a complete reformulation.

### 3. Heuristic Coder Agent
**File:** `src/agents/heuristic_coder.py`  
**Role:** Generate fast, problem-specific greedy/constructive heuristic code  
**Temperature:** 0.7 (allows diversity)

The Heuristic solver generates simple, fast algorithms — greedy construction, rule-based approaches, or domain-specific heuristics. These run quickly and provide a baseline solution.

**Input:** Problem description + formulation + analysis (optional: recommended heuristic strategy)  
**Output:** Self-contained Python script using only stdlib + numpy + scipy

**Code requirements:**
- No commercial solvers (no Gurobi, CPLEX, Pyomo, COPT)
- All problem data embedded as literals in the code
- Must print `OBJECTIVE_VALUE: <number>` as the final output
- Must handle edge cases gracefully

**Why it matters:** In warm-start mode, the heuristic runs first and provides a reference value for the other solvers. Even when inaccurate, it gives meta/hyper solvers a starting point and indicates the problem's approximate scale.

### 4. Metaheuristic Coder Agent
**File:** `src/agents/metaheuristic_coder.py`  
**Role:** Generate population-based or trajectory-based optimization algorithms  
**Temperature:** 0.7

The Metaheuristic solver generates sophisticated optimization algorithms: Genetic Algorithms, Simulated Annealing, Particle Swarm Optimization, Tabu Search, or Differential Evolution. It uses the Analyzer's recommendation to choose the most appropriate algorithm.

**Input:** Problem description + formulation + analysis + warm_start (optional)  
**Output:** Self-contained Python script with the chosen metaheuristic

**Special features:**
- Uses penalty functions or repair operators for constraint handling
- Implements proper convergence criteria
- Must complete within 300 seconds
- If warm-start is provided, treats the heuristic value as a rough reference (explicitly warned it may be wrong)

### 5. Hyperheuristic Coder Agent
**File:** `src/agents/hyperheuristic_coder.py`  
**Role:** Generate adaptive hyper-heuristic frameworks  
**Temperature:** 0.7

The Hyperheuristic solver operates at a meta-level — instead of directly manipulating solutions, it defines 3–5 low-level operators and an adaptive selection mechanism that learns which operators work best during the search.

**Input:** Same as Metaheuristic + recommended operator list from Analyzer  
**Output:** Self-contained Python script with:
- 3–5 low-level operators (greedy construction, local search, perturbation, crossover, etc.)
- Adaptive selection strategy (roulette-wheel with performance weights, RL-style rewards, or choice function)

**Why three solvers?** Each solver type brings different strengths:
- **Heuristic:** Fast, simple, often finds feasible solutions quickly
- **Metaheuristic:** Better optimization quality through systematic search
- **Hyperheuristic:** Most adaptive, can combine strategies dynamically

Having three independent approaches means that even when one fails (wrong algorithm choice, implementation bug), others may succeed. On IndustryOR with GPT-5.4, individual solver accuracy is 67-75%, but the ensemble (any-of-3) achieves 86%.

### 6. Critic Agent
**File:** `src/agents/critic.py`  
**Role:** Pre-execution code review  
**Temperature:** 0.2 (strict)

Before any generated code runs, the Critic reviews it for errors. This catches bugs that would waste execution time and debug retries.

**Review checks:**
1. **Data correctness** — All numerical values match the problem? Typos? Wrong signs?
2. **Constraint implementation** — Inequality directions (≤ vs ≥) correct?
3. **Objective function** — Correct optimization direction? Correct formula?
4. **Numerical issues** — Division by zero? Overflow? Precision problems?
5. **Output format** — Ends with `print(f"OBJECTIVE_VALUE: {value}")`?
6. **Convergence** — Parameters reasonable? Will finish within time limit?
7. **Import restrictions** — Only stdlib + numpy + scipy?

**Output:** Structured review with issues categorized by severity (critical/warning/info). If critical issues are found, the code is sent to the Debugger for fixing before execution.

**Why it matters:** The Critic prevents the most common failure mode — code that crashes on execution. By catching errors before running, it saves expensive compute time and debug retry cycles.

### 7. Debugger Agent
**File:** `src/agents/debugger.py`  
**Role:** Fix broken code using error feedback  
**Temperature:** 0.2 (deterministic fixing)

The Debugger is called in two situations:
1. **After Critic review** — when the Critic identifies critical issues
2. **After execution failure** — when code crashes with an error/traceback

**Input:** Original problem + broken code + error message  
**Output:** Complete fixed Python script

**Fix rules:**
- Return the COMPLETE script (not just changed lines)
- Fix bugs without changing the algorithm approach
- Can adjust parameters or fix logical formulation errors
- Must maintain import restrictions

The Debugger gets up to 3 retry attempts. If all 4 attempts (1 initial + 3 retries) fail, the solver is marked as failed and the pipeline moves on.

### 8. Improver Agent
**File:** `src/agents/improver.py`  
**Role:** Iteratively refine or reformulate when initial solvers fail  
**Temperature:** 0.7 (creative improvements)

The Improver is the most sophisticated agent. It runs after all three solvers have completed, but only if none produced a correct answer. It has **two distinct modes** based on failure pattern detection:

#### Mode 1: REFINE (when solvers disagree or are close to correct)
- Takes the best existing code and improves it
- Fixes mathematical errors
- Increases iterations or compute budget
- Tries different algorithm (SA→GA, GA→DE)
- Improves constraint handling
- Tightens convergence parameters

#### Mode 2: REFORMULATE (when all solvers agree on the same wrong answer)
This is the key architectural breakthrough of v3. When all three independent solvers produce the **exact same wrong answer**, it's almost certainly a formulation error — all solvers correctly solved the wrong problem.

In this mode, the Improver:
- **Ignores ALL previous code and formulation**
- Re-reads the problem word-by-word from scratch
- Identifies likely misunderstandings:
  - Confusing ≤ vs ≥
  - Missing constraints (capacity, budget, precedence)
  - Wrong min/max direction
  - Misread numerical data
  - Ignored integer/binary requirements
- Builds a completely new mathematical model
- Writes a new solver from scratch

**All-same-wrong detection logic:**
```python
# All solvers agree if their values are identical
# or within 0.1% relative error of each other
successful = [r for r in results if r["execution_success"]]
values = [r["objective_value"] for r in successful]
ref = values[0]
all_same = all(abs(v - ref) / abs(ref) < 0.001 for v in values)
```

**Impact:** In v3 with GPT-5.4, the Improver saved 7 additional problems (from 86% to 93%), including one (Problem 21) that required 15 reformulation attempts before finding the correct interpretation.

### 9. Selector Agent
**File:** `src/agents/selector.py`  
**Role:** Ensemble selection — pick the best answer from multiple solvers  
**Temperature:** 0.2

The Selector uses multi-criteria reasoning to choose the single best answer:
1. **Agreement** — If 2+ solvers agree (within 1%), prefer the agreed value
2. **Solver fit** — LP problems → trust heuristic more; combinatorial → trust metaheuristics
3. **Outlier detection** — Wildly different answers are likely wrong
4. **Feasibility** — Successful execution with convergence is more trustworthy
5. **Objective direction** — For minimization, prefer lowest; for maximization, prefer highest

### 10. Reflector Agent
**File:** `src/agents/reflector.py`  
**Role:** Cross-problem learning  
**Temperature:** 0.3

After each problem completes, the Reflector analyzes what happened and extracts generalizable lessons. These lessons are accumulated and shown to the Reflector on subsequent problems, enabling the system to learn patterns within a run.

**Lesson categories:**
- Formulation patterns
- Algorithm selection guidelines
- Parameter tuning insights
- Data handling tips
- Constraint handling strategies
- Convergence observations

**Accumulation:** Lessons are stored in `self._accumulated_lessons` and carried forward across the problem batch. The last 5 lessons are shown to the Reflector for each new problem.

---

## Pipeline Flow

### Phase 1: Analyze
```
Problem (NL) → [Analyzer Agent] → Problem classification + strategy recommendations
```
- Classifies problem type, difficulty, and scale
- Recommends specific algorithms for each solver type
- If analysis fails: continues with empty analysis (non-fatal)

### Phase 2: Formulate
```
Problem (NL) → [Formulator Agent] → Structured mathematical formulation
```
- Extracts objective, variables, constraints, parameters
- Creates shared contract for all downstream agents
- Analysis injected into formulation data for downstream use

### Phase 3: Solve (with Warm-Start)
```
Formulation → [Heuristic] ──(baseline value)──→ [Metaheuristic] ──→ Results
                                              → [Hyperheuristic] ──→ Results
                    ↕                                    ↕
              [Critic] → [Debugger]              [Critic] → [Debugger]
                    ↕                                    ↕
              [Execute + retry]                  [Execute + retry]
```

**Warm-start protocol:**
1. Heuristic runs **alone** first (fast, constructive approach)
2. If heuristic produces a value, it's packaged as warm-start context
3. Metaheuristic and Hyperheuristic run **in parallel**, each receiving the warm-start as a reference point (explicitly marked as potentially incorrect)
4. Each solver's code goes through Critic review → optional Debugger fix → execution with up to 3 debug retries

**Without warm-start:** All three solvers run fully in parallel with no information sharing.

### Phase 4: Improve
```
All Results → [Failure Detection] → REFINE or REFORMULATE
                                        ↓
                              [Improver Agent] → New code
                                        ↓
                              [Execute + retry]
                                        ↓
                              Repeat (up to N iterations)
```

- Only runs if no solver produced a correct answer
- Requires at least one successful execution (needs a baseline to improve)
- Detects "all-same-wrong" pattern → triggers reformulation mode
- Bails out after 3 consecutive execution failures
- Early exit on first correct answer

### Phase 5: Reflect
```
All Results + Lessons → [Reflector Agent] → New lessons → Accumulated for next problem
```

- Runs after every problem (success or failure)
- Extracts lessons about formulation, algorithm selection, etc.
- Lessons accumulate across the problem batch

---

## Data Flow Between Agents

### Information Propagation
```
                    ┌─────────────────┐
                    │   Problem (NL)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    Analyzer     │ → analysis (class, difficulty, recommendations)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Formulator    │ → formulation (objective, variables, constraints)
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
   ┌────────▼────────┐ ┌────▼────┐ ┌────────▼────────┐
   │   Heuristic     │ │  Meta   │ │    Hyper        │
   │ reads: question, │ │ + warm  │ │ + warm_start   │
   │ formulation,    │ │ _start  │ │ + analysis     │
   │ analysis        │ │         │ │ operators      │
   └────────┬────────┘ └────┬────┘ └────────┬────────┘
            │                │                │
            │         ┌─────▼─────┐          │
            ├────────►│  Critic   │◄─────────┤
            │         └─────┬─────┘          │
            │               │ (if issues)    │
            │         ┌─────▼─────┐          │
            ├────────►│ Debugger  │◄─────────┤
            │         └─────┬─────┘          │
            │               │                │
            │         ┌─────▼─────┐          │
            ├────────►│ Executor  │◄─────────┤
            │         └─────┬─────┘          │
            │                │                │
            └────────────────┼────────────────┘
                             │
                    ┌────────▼────────┐
                    │    Improver     │ reads: all results, best code, question
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    Reflector    │ reads: all results, analysis, prior lessons
                    └─────────────────┘
```

### Key Dict Fields at Each Stage

| Stage | Key Fields Read | Key Fields Written |
|-------|----------------|-------------------|
| **Analyzer** | `question` | `analysis`, `analysis_raw` |
| **Formulator** | `question` | `formulation`, `formulation_raw` |
| **Solvers** | `question`, `formulation`, `analysis`, `warm_start` | `solver_type`, `generated_code_raw` |
| **Critic** | `question`, `formulation`, `code`, `solver_type` | `review` (approved, issues) |
| **Debugger** | `question`, `code`, `error` | `fixed_code_raw` |
| **Improver** | `question`, `formulation`, `best_code`, `best_value`, `all_results`, `iteration` | `generated_code_raw`, `improvement_mode` |
| **Reflector** | `question`, `analysis`, `results`, `is_correct`, `prior_lessons` | `reflection` (lessons, solver_performance) |

---

## Execution and Error Recovery

### Code Execution
All generated Python code runs in an isolated subprocess via `src/execution/sandbox.py`:
- Temporary `.py` file written to disk
- Executed with `subprocess.run()` with configurable timeout (default: 600s)
- Stdout parsed for `OBJECTIVE_VALUE: <number>` pattern
- Success = exit code 0 AND objective value found

### Debug Retry Loop
Each solver attempt follows this loop:

```
Generate code → Critic review → [fix if needed] → Execute
                                                      │
                                    ┌─────────── fail ─┘
                                    │
                              Debugger fix → Execute (retry 1)
                                                      │
                                    ┌─────────── fail ─┘
                                    │
                              Debugger fix → Execute (retry 2)
                                                      │
                                    ┌─────────── fail ─┘
                                    │
                              Debugger fix → Execute (retry 3)
                                                      │
                                    └─── fail → Mark as failed
```

### Improvement Loop Bailout
The Improver has a safety mechanism: if 3 consecutive improvement iterations produce code that fails to execute, the loop terminates. This prevents spending hours on problems where the model fundamentally cannot produce working code.

---

## Parallelism Architecture

### Problem-Level Parallelism
Multiple problems can be solved concurrently using `--parallel-problems N`. Each problem runs the full pipeline independently in a `ThreadPoolExecutor`. Results are accumulated under a thread-safe lock.

### Solver-Level Parallelism
Within each problem, solvers can run in parallel:
- **Warm-start mode:** Heuristic first (sequential), then Meta + Hyper in parallel
- **Non-warm-start:** All three solvers in parallel
- **Sequential mode (`--sequential`):** All solvers one at a time (required for some local models like gemma4:26b to avoid Ollama crashes)

### Langfuse Trace Propagation
When running parallel threads, the orchestrator captures the current Langfuse trace ID and observation ID from the parent thread and passes them to each child thread. This ensures parallel solver executions are properly grouped under a single trace hierarchy in the Langfuse UI.

---

## Configuration Flags

Each phase of the pipeline can be independently toggled:

| Flag | Default | Effect |
|------|---------|--------|
| `--no-analyze` | Enabled | Skip Problem Analyzer |
| `--no-warm-start` | Enabled | Run all solvers in parallel (no heuristic-first) |
| `--no-critic` | Enabled | Skip pre-execution code review |
| `--improve-iterations N` | 2 | Number of improvement iterations (0 to disable) |
| `--no-selector` | Enabled | Skip smart ensemble selection |
| `--no-reflector` | Enabled | Skip cross-problem learning |
| `--legacy` | Disabled | Disable ALL multi-agent enhancements (equivalent to v1) |
| `--sequential` | Disabled | Run solvers sequentially (required for some Ollama models) |
| `--parallel-problems N` | 4 | Number of problems to solve concurrently |

---

## Impact of the Multi-Agent Architecture

### v1 (Baseline) vs v3 (Multi-Agent)

| Component | v1 | v3 | Impact |
|-----------|-----|-----|--------|
| Analyzer | ❌ | ✅ | Better algorithm selection |
| Formulator | ✅ | ✅ | — |
| Solvers (×3) | ✅ parallel | ✅ warm-start | Heuristic seeds meta/hyper |
| Critic | ❌ | ✅ | Catches bugs pre-execution |
| Debugger | ✅ (3 retries) | ✅ (3 retries) | — |
| Improver | ❌ | ✅ (reformulate + refine) | **+7 problems** via reformulation |
| Selector | ❌ | ✅ | Smart ensemble selection |
| Reflector | ❌ | ✅ | Cross-problem learning |

### Accuracy Progression (IndustryOR, GPT-5.4)

| Version | Ensemble Accuracy | Key Change |
|---------|------------------|------------|
| v1 (baseline) | 81% | 3-solver parallel ensemble |
| v2 (first multi-agent) | 80% | Added agents but warm-start biased hyper |
| **v3 (targeted fixes)** | **93%** | Reformulation improver + soft warm-start |

### Model Scaling (v3 Architecture)

| Model | Ensemble | Notes |
|-------|----------|-------|
| GPT-5.4 (cloud) | 93% | Full capability |
| GPT-4o (cloud) | 61% | Architecture adds +23pp over v1 |
| Llama 3 8B (local) | 0% | Below minimum capability threshold |

The multi-agent architecture provides the greatest benefit to mid-range models — it adds +23pp to GPT-4o (38→61%) but only +12pp to GPT-5.4 (81→93%). For models below a minimum capability threshold (Llama 3 8B), the architecture cannot compensate for the inability to generate correct code.
