# ll4or — Multi-Agent Heuristic Solver for Operations Research

A multi-agent system that takes OR problems from benchmark datasets, uses LLM agents to generate **heuristic**, **metaheuristic**, and **hyper-heuristic** Python code, executes it, and evaluates results against ground truth.

## Quick Start

```bash
# Set up your LLM provider in .env (see .env.example)
# Then run:
python3 -m src.main --dataset industryOR --max-problems 5 --timeout 120

# Run a specific solver type
python3 -m src.main --dataset bwor --solver metaheuristic --max-problems 3

# List available datasets
python3 -m src.main --list-datasets
```

## Architecture

```
Dataset ─→ Formulator Agent ─→ Solver Agents (×3) ─→ Executor ─→ Evaluator
              (NL → math)     heuristic / meta /    (subprocess)  (per-dataset
                              hyper-heuristic                      comparison)
                                    │
                              Debugger Agent ←── on failure (up to 3 retries)
```

All generated code is **pure Python** (stdlib + numpy/scipy) — no commercial solvers required.

## Datasets

| Name | Key | Problems | Description |
|------|-----|----------|-------------|
| IndustryOR | `industryOR` | 100 | Real-world industrial OR (LP, IP, MIP, NLP) |
| BWOR | `bwor` | 82 | High-quality curated OR problems (EN + CN) |
| MAMO Easy | `mamo_easy` | 652 | Straightforward linear programming |
| MAMO Complex | `mamo_complex` | 211 | Multi-constraint LP problems |

## Evaluation Methodology

Each dataset uses a **different evaluation method** matching its benchmark's reference implementation. Our system automatically selects the correct method per dataset.

### IndustryOR / ORLM — Relative Error with Integer Rounding

**Reference:** [`datasets/ORLM/eval/execute.py`](datasets/ORLM/eval/execute.py)

| Property | Value |
|----------|-------|
| **Mode** | Relative error |
| **Tolerance** | 5% (`\|pred − gt\| / \|gt\| ≤ 0.05`) |
| **Rounding** | Both values rounded to **integers** before comparison |
| **GT = 0** | Absolute error ≤ 0.05 |
| **Infeasible** | String match on `"No Best Solution"` |
| **Metrics** | pass@k, majority voting (MJ@k) |

```
Example: predicted=220080, gt=219816
  → round(220080)=220080, round(219816)=219816
  → |220080 − 219816| / 219816 = 0.12%  ✓ (within 5%)
```

### BWOR — Absolute Error

**Reference:** [`datasets/BWOR/utils.py`](datasets/BWOR/utils.py) → `eval_model_result()`

| Property | Value |
|----------|-------|
| **Mode** | Absolute error |
| **Tolerance** | 0.1 (`\|pred − gt\| < 0.1`) |
| **Rounding** | None — raw float comparison |
| **Infeasible** | `None` or `"None"` matches null ground truth |
| **Metrics** | pass_flag (executed), correct_flag (within tolerance) |

```
Example: predicted=5450.05, gt=5450.0
  → |5450.05 − 5450.0| = 0.05  ✓ (< 0.1)

Example: predicted=5700.0, gt=5450.0
  → |5700.0 − 5450.0| = 250.0  ✗ (≥ 0.1, even though 4.6% relative)
```

> **Key difference from ORLM:** A result that is 4.6% off would pass ORLM's 5% relative check but fail BWOR's 0.1 absolute check. BWOR requires near-exact numerical answers.

### MAMO (Easy LP & Complex LP) — Hybrid Comparison

**Reference:** [`datasets/MAMO/scripts/scripts_optimization/mamo_script_optimization/3.run_code_comp_optimization.py`](datasets/MAMO/) → `comp()` + `compare_output_with_standard()`

| Property | Value |
|----------|-------|
| **Mode** | Hybrid: scale-based decimal check **OR** relative error |
| **Tolerance** | 0.01% relative (`\|pred − gt\| / \|gt\| ≤ 1e-4`) |
| **Scale check** | Multiply both by `10^(decimal digits)`, check `\|diff\| < 1` |
| **Rounding** | None — raw float comparison |
| **Metrics** | Accuracy = correct / total |

The hybrid approach works in two steps — **either** passing means correct:

1. **Scale-based:** Determine decimal digits from ground truth string. Multiply both values by `10^digits` (min 2). If `|scaled_pred − scaled_gt| < 1`, it's correct.
2. **Relative:** If `|pred − gt| / |gt| ≤ 1e-4`, it's correct.

```
Example: predicted=10000.5, gt=10000 (raw string "10000")
  → Scale: 10000.5 × 100 − 10000 × 100 = 50  ✗ (≥ 1)
  → Relative: |0.5| / 10000 = 0.00005  ✓ (≤ 1e-4)
  → Result: ✓ (relative check passed)

Example: predicted=3.1416, gt=3.1415 (raw string "3.1415")
  → Scale: 3.1416 × 10000 − 3.1415 × 10000 = 1.0  ✗ (≥ 1, borderline)
  → Relative: |0.0001| / 3.1415 = 0.000032  ✓ (≤ 1e-4)
  → Result: ✓
```

> **Key difference:** MAMO is ~50× stricter than ORLM. A result with 1% error passes ORLM easily but fails MAMO.

### Comparison Summary

| Dataset | Type | Tolerance | Strictness | Rounding |
|---------|------|-----------|------------|----------|
| **ORLM / IndustryOR** | Relative | 5% | Lenient | Integer |
| **BWOR** | Absolute | 0.1 | Near-exact | None |
| **MAMO** | Hybrid | 0.01% | Very strict | None |

### Datasets Without Numerical Evaluation

| Dataset | Evaluation Type | Notes |
|---------|----------------|-------|
| **NL4Opt** | Structural (NER F1 / LP formulation accuracy) | Evaluates problem *parsing*, not solution accuracy |
| **OptiMUS** | Code generation pass/fail | Checks if generated code runs, not answer correctness |
| **ORQA** | Multiple-choice QA accuracy | Standard MCQ evaluation |

## Project Structure

```
src/
├── llm/                  # LLM client abstraction
│   ├── base.py           #   Abstract LLMClient interface
│   ├── openai_client.py  #   OpenAI API
│   ├── azure_client.py   #   Azure OpenAI API
│   └── anthropic_client.py # Anthropic API
├── agents/               # LLM-powered agents
│   ├── formulator.py     #   NL → structured math formulation
│   ├── heuristic_coder.py    # Greedy/constructive heuristics
│   ├── metaheuristic_coder.py # GA, SA, PSO, Tabu Search
│   ├── hyperheuristic_coder.py # Adaptive operator selection
│   └── debugger.py       #   Error-driven code repair
├── datasets/             # Dataset adapters (decoupled from solver)
│   ├── base.py           #   DatasetAdapter ABC + Problem dataclass
│   ├── orlm.py           #   ORLM / IndustryOR
│   ├── bwor.py           #   BWOR benchmark
│   ├── mamo.py           #   MAMO Easy/Complex LP
│   └── registry.py       #   Dataset name → adapter lookup
├── execution/
│   └── sandbox.py        # Subprocess code execution with timeout
├── evaluation/
│   └── evaluator.py      # Per-dataset comparison + aggregate metrics
├── orchestrator.py       # Pipeline wiring
├── config.py             # Configuration (from .env)
└── main.py               # CLI entry point
```

## Configuration

Copy `.env` and set your LLM credentials:

```bash
LLM_PROVIDER=azure          # openai | anthropic | azure
LLM_MODEL=gpt-5.4
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_API_VERSION=2025-04-01-preview
EXEC_TIMEOUT=600
OUTPUT_DIR=results
```

## Observability with Langfuse

The pipeline integrates with [Langfuse](https://langfuse.com/) for full LLM observability — traces every agent call with prompt, completion, token usage, latency, and cost.

### Start Langfuse (local, self-hosted)

```bash
# Start the Langfuse stack (Postgres, ClickHouse, Redis, MinIO, Web, Worker)
docker compose -f docker-compose.langfuse.yml up -d

# Wait ~30 seconds for initialization, then open:
open http://localhost:3000
```

**Login credentials:**

| Field | Value |
|-------|-------|
| Email | `admin@local.dev` |
| Password | `adminadmin` |

A default project ("OR Solver") is auto-created with API keys that match the `.env` file — no manual setup needed.

### What gets traced

Every pipeline run creates a hierarchical trace in Langfuse:

```
pipeline_run (dataset, accuracy, per-solver metrics)
  └─ solve_problem (per problem)
       ├─ formulator LLM call (prompt, completion, tokens, latency)
       ├─ run_solver: heuristic
       │    └─ LLM call (code generation)
       │    └─ [debugger LLM call, if retry needed]
       ├─ run_solver: metaheuristic
       │    └─ LLM call
       └─ run_solver: hyperheuristic
            └─ LLM call
```

### Configuration

In `.env`:

```bash
LANGFUSE_ENABLED=true              # set to false to disable (pipeline runs normally)
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-ll4or-local
LANGFUSE_SECRET_KEY=sk-ll4or-local
```

### Stop / Reset

```bash
# Stop Langfuse (data preserved)
docker compose -f docker-compose.langfuse.yml down

# Stop and delete all data
docker compose -f docker-compose.langfuse.yml down -v
```

## Adding a New Dataset

1. Create a new adapter in `src/datasets/` implementing `DatasetAdapter`:
   - `load()` — parse your data files
   - `get_problems()` — return `list[Problem]`
   - `get_eval_config()` — return the correct `EvaluationConfig` for your benchmark
2. Register it in `src/datasets/registry.py`

The solver pipeline is fully decoupled — it only sees `Problem(question, answer)` objects and never touches raw dataset files.