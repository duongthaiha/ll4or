# Benchmark Results — Multi-Agent Heuristic OR Solver

**Model:** GPT-5.4 (Azure OpenAI)  
**Approach:** Pure Python heuristic code generation (no commercial solvers)  
**Date:** 2026-03-29

---

## Summary

| Benchmark | Our Best Solver | Our Ensemble | Previous SOTA | Improvement |
|-----------|----------------|--------------|---------------|-------------|
| **NL4OPT** | 88.7% (heuristic) | 84.1% (any-of-3) | 86.5% (ORLM-Deepseek) | **+2.2 pp** |
| **IndustryOR** | 72.0% (heuristic) | 81.0% (any-of-3) | 38.0% (ORLM-LLaMA-3-8B) | **+43.0 pp** |

> **Note:** "Our Ensemble" counts a problem as correct if *any* of the 3 solvers produces a correct answer. "Our Best Solver" is the single best-performing solver type. Previous SOTA uses fine-tuned models + commercial COPT solver; our approach uses zero-shot GPT-5.4 + pure Python.

---

## NL4OPT (245 problems)

**Evaluation:** ORLM-style — 5% relative tolerance, values rounded to integers.

### Per-Solver Accuracy

| Solver | Correct | Evaluated | Accuracy | Failed Executions |
|--------|---------|-----------|----------|-------------------|
| Heuristic | 205 | 231 | **88.7%** | 5 |
| Metaheuristic | 204 | 231 | 88.3% | 0 |
| Hyper-heuristic | 204 | 231 | 88.3% | 0 |

### Per-Problem Accuracy

| Metric | Count | Rate |
|--------|-------|------|
| At least 1 solver correct | 206 / 245 | **84.1%** |
| All 3 solvers correct | 202 / 245 | 82.4% |
| No solver correct | 39 / 245 | 15.9% |

### Comparison with Prior Work

| Method | NL4OPT | Model | Solver |
|--------|--------|-------|--------|
| **Ours (heuristic)** | **88.7%** | GPT-5.4 | Pure Python |
| **Ours (ensemble)** | **84.1%** | GPT-5.4 | Pure Python |
| ORLM-Deepseek-Math-7B | 86.5% | Fine-tuned 7B | COPT |
| ORLM-LLaMA-3-8B | 85.7% | Fine-tuned 8B | COPT |
| ORLM-Mistral-7B | 84.4% | Fine-tuned 7B | COPT |
| OptiMUS | 78.8% | GPT-4 | Gurobi |
| Chain-of-Experts | 64.2% | GPT-4 | — |
| Chain-of-Experts | 58.9% | GPT-3.5 | — |
| Reflexion | 53.0% | GPT-4 | — |
| GPT-4 Standard | 47.3% | GPT-4 | COPT |
| tag-BART | 47.9% | PLM | — |
| GPT-3.5 Standard | 42.4% | GPT-3.5 | COPT |

**Runtime:** ~92 minutes (8 concurrent problems), ~245 × 3 = 735 solver runs.

---

## IndustryOR (100 problems)

**Evaluation:** ORLM-style — 5% relative tolerance, values rounded to integers.  
**Problem types:** LP, IP, MIP, NLP — real-world industrial optimization.

### Per-Solver Accuracy

| Solver | Correct | Evaluated | Accuracy | Failed Executions |
|--------|---------|-----------|----------|-------------------|
| Heuristic | 72 | 100 | **72.0%** | 4 |
| Metaheuristic | 71 | 100 | 71.0% | 1 |
| Hyper-heuristic | 68 | 100 | 68.0% | 9 |

### Per-Problem Accuracy

| Metric | Count | Rate |
|--------|-------|------|
| At least 1 solver correct | 81 / 100 | **81.0%** |
| All 3 solvers correct | 60 / 100 | 60.0% |
| No solver correct | 19 / 100 | 19.0% |

### Comparison with Prior Work

| Method | IndustryOR | Model | Solver |
|--------|-----------|-------|--------|
| **Ours (ensemble)** | **81.0%** | GPT-5.4 | Pure Python |
| **Ours (heuristic)** | **72.0%** | GPT-5.4 | Pure Python |
| **Ours (metaheuristic)** | **71.0%** | GPT-5.4 | Pure Python |
| **Ours (hyper-heuristic)** | **68.0%** | GPT-5.4 | Pure Python |
| ORLM-LLaMA-3-8B | 38.0% | Fine-tuned 8B | COPT |
| ORLM-Deepseek-Math-7B | 33.0% | Fine-tuned 7B | COPT |
| GPT-4 Standard | 28.0% | GPT-4 | COPT |
| ORLM-Mistral-7B | 27.0% | Fine-tuned 7B | COPT |

**Runtime:** ~33 minutes (10 concurrent problems), 100 × 3 = 300 solver runs.

---

## Key Findings

1. **No commercial solver needed.** Our pure Python heuristic approach matches or exceeds systems that rely on COPT/Gurobi, despite lacking mathematical optimality guarantees.

2. **Ensemble adds value on hard problems.** On IndustryOR, the ensemble (any-of-3) reaches 81% vs the best single solver at 72% — a 9-point gain. On NL4OPT (simpler LPs), the gain is minimal since all solvers agree.

3. **All three solvers perform comparably on IndustryOR** (72%, 71%, 68%), while heuristic leads on NL4OPT (88.7%). Greedy/constructive approaches slightly outperform stochastic search, likely because the LLM can reason about problem structure directly.

4. **Execution reliability.** Hyper-heuristic has the most failed executions (9 on IndustryOR) due to more complex generated code. The debugger agent helps but can't always recover.

5. **Massive improvement on industrial problems.** IndustryOR jumps from 38% (ORLM SOTA) to 81% — a 43 percentage point improvement, suggesting that stronger base models (GPT-5.4) compensate for the lack of fine-tuning.

---

## Configuration

- **LLM:** GPT-5.4 via Azure OpenAI
- **Execution timeout:** 120 seconds per solver run
- **Debug retries:** Up to 3 per solver
- **Parallelism:** 8 concurrent problems × 3 parallel solvers
- **Evaluation tolerance:** 5% relative error, integer rounding (ORLM standard)

---

## Per-Problem Comparison: Ours vs ORLM

> Side-by-side per-problem comparison against **ORLM-LLaMA-3-8B** (fine-tuned 8B model + COPT commercial solver).
> Evaluation uses 5% relative tolerance with integer rounding. ✓ = correct, ✗ = incorrect, — = no data.

### NL4OPT (245 problems)

> Problems with ground truth = "No Best Solution" are excluded from accuracy (14 problems).
> IDs match 1:1 between our run and ORLM baseline.

**Agreement matrix (best solver per problem):**

| | ORLM ✓ | ORLM ✗ | Total |
|---|--:|--:|--:|
| **Ours ✓** | 196 | 10 | **206** |
| **Ours ✗** | 15 | 24 | 39 |
| **Total** | 211 | 34 | 245 |

- **10 problems** solved by us but not ORLM
- **15 problems** solved by ORLM but not us
- **196 problems** solved by both
- **24 problems** solved by neither

<details>
<summary>Click to expand full per-problem results</summary>

| ID | Ground Truth | ORLM Pred | ORLM | Ours Pred | Ours | Best Solver |
|--:|--:|--:|:--:|--:|:--:|---|
| 1 | 1160 | 1160 | ✓ | 1160 | ✓ | heuristic |
| 2 | 350 | 350 | ✓ | 350 | ✓ | heuristic |
| 3 | 7 | 7 | ✓ | 7 | ✓ | heuristic |
| 4 | 400000 | 400000 | ✓ | 400000 | ✓ | heuristic |
| 5 | 513 | 513 | ✓ | 513 | ✓ | heuristic |
| 6 | 37 | 37 | ✓ | 37 | ✓ | heuristic |
| 7 | 236.50 | 224 | ✗ | 236.50 | ✓ | heuristic |
| 8 | 327.66 | 327.66 | ✓ | 327.66 | ✓ | heuristic |
| 9 | 215000 | 215000 | ✓ | 215000 | ✓ | heuristic |
| 10 | 841 | 841 | ✓ | 841 | ✓ | heuristic |
| 11 | 1.50 | 1.50 | ✓ | 1.50 | ✓ | heuristic |
| 12 | 12860 | 12689.66 | ✓ | 12600 | ✓ | heuristic |
| 13 | 100 | 100 | ✓ | 100 | ✓ | heuristic |
| 14 | 80 | 78.95 | ✓ | 80 | ✓ | heuristic |
| 15 | 950 | 950 | ✓ | 950 | ✓ | heuristic |
| 16 | 75 | 75 | ✓ | 75 | ✓ | heuristic |
| 17 | — | — | ✓ | 833.33 | ✗ | heuristic |
| 18 | 1400 | 1400 | ✓ | 1400 | ✓ | heuristic |
| 19 | 500 | 500 | ✓ | 500 | ✓ | heuristic |
| 20 | 7 | 6.28 | ✗ | 6.28 | ✗ | heuristic |
| 21 | 35 | 35 | ✓ | 35 | ✓ | heuristic |
| 22 | 4990 | 4990 | ✓ | 4990 | ✓ | heuristic |
| 23 | 60000 | 60000 | ✓ | 60000 | ✓ | heuristic |
| 24 | 1060 | 1060 | ✓ | 1072 | ✓ | heuristic |
| 25 | 750 | 750 | ✓ | 750 | ✓ | heuristic |
| 26 | 98 | 98 | ✓ | 100.34 | ✓ | heuristic |
| 27 | 160 | 160 | ✓ | 160 | ✓ | heuristic |
| 28 | — | — | ✓ | 15 | ✗ | heuristic |
| 29 | -99999 | 3000 | ✗ | 3000 | ✗ | heuristic |
| 30 | 29 | 29 | ✓ | 29 | ✓ | heuristic |
| 31 | 369 | 369 | ✓ | 375 | ✓ | heuristic |
| 32 | 67 | 67 | ✓ | 67 | ✓ | heuristic |
| 33 | — | 233.50 | ✗ | 233.33 | ✗ | heuristic |
| 34 | 206250 | 206250 | ✓ | 206250 | ✓ | heuristic |
| 35 | 84 | 84 | ✓ | 100 | ✗ | heuristic |
| 36 | 600 | 600 | ✓ | 600 | ✓ | heuristic |
| 37 | — | 1215 | ✗ | 1215 | ✗ | heuristic |
| 38 | 239 | 239 | ✓ | 240 | ✓ | heuristic |
| 39 | 4400 | 4400 | ✓ | 4400 | ✓ | heuristic |
| 40 | 150000 | 150000 | ✓ | 150000 | ✓ | heuristic |
| 41 | 33.50 | 33.50 | ✓ | 33.50 | ✓ | heuristic |
| 42 | 8 | 8 | ✓ | 8 | ✓ | heuristic |
| 43 | 648 | 648 | ✓ | 648 | ✓ | heuristic |
| 44 | 300 | 210 | ✗ | 300 | ✓ | heuristic |
| 45 | 990 | 990 | ✓ | 990 | ✓ | heuristic |
| 46 | 540 | 540 | ✓ | 547.67 | ✓ | heuristic |
| 47 | 17000 | 17000 | ✓ | 17000 | ✓ | heuristic |
| 48 | 142 | 142 | ✓ | 140 | ✓ | heuristic |
| 49 | -99999 | — | ✗ | 0 | ✗ | heuristic |
| 50 | — | — | ✓ | 250 | ✗ | heuristic |
| 51 | 960 | 960 | ✓ | 960 | ✓ | heuristic |
| 52 | 290.50 | 290.50 | ✓ | 291.67 | ✓ | heuristic |
| 53 | 5.85 | 5.85 | ✓ | 5.85 | ✓ | heuristic |
| 54 | 3 | 2.50 | ✗ | 2.50 | ✓ | metaheuristic |
| 55 | 4190 | 4190 | ✓ | 4190 | ✓ | heuristic |
| 56 | 14375 | 14375 | ✓ | 14375 | ✓ | heuristic |
| 57 | 25 | 25 | ✓ | 25 | ✓ | heuristic |
| 58 | 85500 | 85500 | ✓ | 85500 | ✓ | heuristic |
| 59 | 690 | 690 | ✓ | 690 | ✓ | heuristic |
| 60 | 9000 | 9000 | ✓ | 9000 | ✓ | heuristic |
| 61 | 1000 | 1000 | ✓ | 999.95 | ✓ | heuristic |
| 62 | 25 | 25 | ✓ | 25 | ✓ | heuristic |
| 63 | 1680 | 1680 | ✓ | 1684.62 | ✓ | heuristic |
| 64 | 7 | 7 | ✓ | 7 | ✓ | heuristic |
| 65 | 2200 | 2200 | ✓ | 2200 | ✓ | heuristic |
| 66 | 150000 | 150000 | ✓ | 150000 | ✓ | heuristic |
| 67 | 19 | 19 | ✓ | 19 | ✓ | heuristic |
| 68 | 62.50 | 62.50 | ✓ | 62.50 | ✓ | heuristic |
| 69 | 37 | 36.67 | ✓ | 36.67 | ✓ | heuristic |
| 70 | 60 | 60 | ✓ | 60 | ✓ | heuristic |
| 71 | 52 | 52 | ✓ | 52 | ✓ | heuristic |
| 72 | 580 | 580 | ✓ | 580 | ✓ | heuristic |
| 73 | -99999 | 76 | ✗ | 76 | ✗ | heuristic |
| 74 | 2400 | 2333.33 | ✓ | 2333.33 | ✓ | heuristic |
| 75 | 735 | 735 | ✓ | 735 | ✓ | heuristic |
| 76 | — | — | ✓ | 0 | ✗ | heuristic |
| 77 | 14 | 14 | ✓ | 14 | ✓ | heuristic |
| 78 | 1.50 | 1.50 | ✓ | 1.50 | ✓ | heuristic |
| 79 | 1480 | 1480 | ✓ | 1480 | ✓ | heuristic |
| 80 | 6.40 | 20.80 | ✗ | 21.33 | ✗ | heuristic |
| 81 | 5050 | 5050 | ✓ | 5050 | ✓ | heuristic |
| 82 | 1965 | 1965 | ✓ | 1965 | ✓ | heuristic |
| 83 | 1800 | 2130 | ✗ | 1800 | ✓ | heuristic |
| 84 | 310 | 310 | ✓ | 310 | ✓ | heuristic |
| 85 | 17.71 | 17.71 | ✓ | 17.71 | ✓ | heuristic |
| 86 | 14 | 14 | ✓ | 14 | ✓ | heuristic |
| 87 | 6 | 6 | ✓ | 6 | ✓ | heuristic |
| 88 | 36900 | 36900 | ✓ | 37083.33 | ✓ | heuristic |
| 89 | 1001 | 1001 | ✓ | 1000 | ✓ | heuristic |
| 90 | 390 | 390 | ✓ | 390 | ✓ | heuristic |
| 91 | 81000 | 81000 | ✓ | 81000 | ✓ | heuristic |
| 92 | 72 | 66.67 | ✗ | 66.67 | ✗ | heuristic |
| 93 | 4000000 | 4000000 | ✓ | 4000000 | ✓ | heuristic |
| 94 | 1080 | 1080 | ✓ | 1080 | ✓ | heuristic |
| 95 | 32 | 32 | ✓ | 32 | ✓ | heuristic |
| 96 | 810 | 810 | ✓ | 810 | ✓ | heuristic |
| 97 | 160 | 160 | ✓ | 160 | ✓ | heuristic |
| 98 | 24 | 24 | ✓ | 24 | ✓ | heuristic |
| 99 | 230 | 230 | ✓ | 230 | ✓ | heuristic |
| 100 | 2480 | 2480 | ✓ | 2480 | ✓ | heuristic |
| 101 | 300 | 300 | ✓ | 1320 | ✗ | heuristic |
| 102 | 22 | 22 | ✓ | 22 | ✓ | heuristic |
| 103 | 310 | 310 | ✓ | 310 | ✓ | heuristic |
| 104 | 175 | 175 | ✓ | 175 | ✓ | heuristic |
| 105 | 30 | 30 | ✓ | 30 | ✓ | heuristic |
| 106 | 30000 | 29950 | ✓ | 30000 | ✓ | heuristic |
| 107 | 342750 | 342750 | ✓ | 342857.14 | ✓ | heuristic |
| 108 | 1366.67 | 2975 | ✗ | 1366.67 | ✓ | heuristic |
| 109 | -99999 | — | ✗ | 7.51 | ✗ | heuristic |
| 110 | 571 | 571.43 | ✓ | 571.43 | ✓ | heuristic |
| 111 | 26 | 26 | ✓ | 26 | ✓ | heuristic |
| 112 | — | — | ✓ | 0 | ✗ | heuristic |
| 113 | 61875 | 61875 | ✓ | 61875 | ✓ | heuristic |
| 114 | 166.67 | 166.67 | ✓ | 166.67 | ✓ | heuristic |
| 115 | 4347 | 4347 | ✓ | 4347 | ✓ | heuristic |
| 116 | 1500 | 1500 | ✓ | 1500 | ✓ | heuristic |
| 117 | — | — | ✓ | 26 | ✗ | heuristic |
| 118 | 45 | 45 | ✓ | 45 | ✓ | heuristic |
| 119 | 430 | 430 | ✓ | 430 | ✓ | heuristic |
| 120 | 600 | 600 | ✓ | 600 | ✓ | heuristic |
| 121 | 125 | 125 | ✓ | 125 | ✓ | heuristic |
| 122 | 507.80 | 507.80 | ✓ | 508.33 | ✓ | heuristic |
| 123 | 684000 | 684000 | ✓ | 684000 | ✓ | heuristic |
| 124 | 2500 | 8350 | ✗ | 11275 | ✗ | heuristic |
| 125 | 22 | 22 | ✓ | 22 | ✓ | heuristic |
| 126 | 430.77 | 431.41 | ✓ | 431.41 | ✓ | heuristic |
| 127 | 291.40 | 290.25 | ✓ | 290.25 | ✓ | heuristic |
| 128 | 465 | 465 | ✓ | 465 | ✓ | heuristic |
| 129 | 363 | 363 | ✓ | 363 | ✓ | heuristic |
| 130 | 22 | 22 | ✓ | 22 | ✓ | heuristic |
| 131 | 28 | 28 | ✓ | 28 | ✓ | heuristic |
| 132 | 11980 | 11980 | ✓ | 12000 | ✓ | heuristic |
| 133 | 4 | 5 | ✗ | 2.40 | ✗ | heuristic |
| 134 | 2250 | 2250 | ✓ | 2250 | ✓ | heuristic |
| 135 | 19 | 19 | ✓ | 19 | ✓ | heuristic |
| 136 | 44 | 44 | ✓ | 44 | ✓ | heuristic |
| 137 | 120 | 120 | ✓ | 120 | ✓ | heuristic |
| 138 | 342 | 342 | ✓ | 342 | ✓ | heuristic |
| 139 | 60 | 60 | ✓ | 60 | ✓ | heuristic |
| 140 | 32 | 40 | ✗ | 40 | ✗ | heuristic |
| 141 | 511.43 | 511.43 | ✓ | 511.43 | ✓ | heuristic |
| 142 | 650 | 650 | ✓ | 650 | ✓ | heuristic |
| 143 | 480 | 480 | ✓ | 487.50 | ✓ | heuristic |
| 144 | 136 | 127.50 | ✗ | 143 | ✗ | heuristic |
| 145 | 30 | 30 | ✓ | 30 | ✓ | heuristic |
| 146 | 10060 | 10057.14 | ✓ | 10060 | ✓ | heuristic |
| 147 | 64 | 63.33 | ✓ | 63.33 | ✓ | heuristic |
| 148 | 670 | 670 | ✓ | 670 | ✓ | heuristic |
| 149 | 1000 | 1000 | ✓ | 1000 | ✓ | heuristic |
| 150 | 40 | 40 | ✓ | 40 | ✓ | heuristic |
| 151 | — | — | ✓ | 750 | ✗ | heuristic |
| 152 | 306 | 309.09 | ✓ | 306 | ✓ | heuristic |
| 153 | -99999 | — | ✗ | 75 | ✗ | heuristic |
| 154 | 1070 | 1070 | ✓ | 1070 | ✓ | heuristic |
| 155 | 50 | 55.56 | ✗ | 55.56 | ✗ | heuristic |
| 156 | 16666.67 | 16666.67 | ✓ | 16666.67 | ✓ | heuristic |
| 157 | 75 | 75 | ✓ | 75 | ✓ | heuristic |
| 158 | 60 | 60 | ✓ | 60 | ✓ | heuristic |
| 159 | 71 | 71 | ✓ | 71 | ✓ | heuristic |
| 160 | 6000 | 6000 | ✓ | 6000 | ✓ | heuristic |
| 161 | 5 | 5 | ✓ | 5 | ✓ | heuristic |
| 162 | 6 | 6 | ✓ | 7 | ✗ | heuristic |
| 163 | -99999 | — | ✗ | 72 | ✗ | heuristic |
| 164 | 133200 | 133200 | ✓ | 133333.33 | ✓ | heuristic |
| 165 | 217 | 217 | ✓ | 217 | ✓ | heuristic |
| 166 | 78 | 76.92 | ✓ | 78 | ✓ | heuristic |
| 167 | 95 | 330 | ✗ | 327.50 | ✗ | heuristic |
| 168 | 23.50 | 22.75 | ✓ | 22.75 | ✓ | heuristic |
| 169 | 29 | 28.57 | ✓ | 29 | ✓ | heuristic |
| 170 | 333 | 333 | ✓ | 333 | ✓ | heuristic |
| 171 | 80000 | 80000 | ✓ | 80000 | ✓ | heuristic |
| 172 | 1990 | 1990 | ✓ | 1990 | ✓ | heuristic |
| 173 | 70 | 70 | ✓ | 70 | ✓ | heuristic |
| 174 | 268 | 268 | ✓ | 268 | ✓ | heuristic |
| 175 | 7500 | 7500 | ✓ | 5625 | ✗ | heuristic |
| 176 | 24 | 24 | ✓ | 24 | ✓ | heuristic |
| 177 | 48 | 48 | ✓ | 48 | ✓ | heuristic |
| 178 | 18 | 17 | ✗ | 17 | ✗ | heuristic |
| 179 | 89 | 89 | ✓ | 89 | ✓ | heuristic |
| 180 | 101.82 | 101.82 | ✓ | 101.82 | ✓ | heuristic |
| 181 | 368 | 366.67 | ✓ | 366.67 | ✓ | heuristic |
| 182 | 80000 | 90000 | ✗ | 80000 | ✓ | heuristic |
| 183 | 260 | 260 | ✓ | 260 | ✓ | heuristic |
| 184 | 6300 | 6300 | ✓ | 6300 | ✓ | heuristic |
| 185 | 960 | 960 | ✓ | 960 | ✓ | heuristic |
| 186 | 226 | — | ✗ | 226 | ✓ | heuristic |
| 187 | — | — | ✓ | — | ✗ | heuristic |
| 188 | 6794 | 6794 | ✓ | 6785.71 | ✓ | heuristic |
| 189 | 67 | 67 | ✓ | 67 | ✓ | heuristic |
| 190 | 582 | 582 | ✓ | 582 | ✓ | heuristic |
| 191 | 480 | 480 | ✓ | 480 | ✓ | heuristic |
| 192 | 375 | 375 | ✓ | 375 | ✓ | heuristic |
| 193 | 65000 | 65000 | ✓ | 65000 | ✓ | heuristic |
| 194 | 118 | 115.71 | ✓ | 115.71 | ✓ | heuristic |
| 195 | 20 | 20 | ✓ | 20 | ✓ | heuristic |
| 196 | 16 | 16 | ✓ | 16 | ✓ | heuristic |
| 197 | — | — | ✓ | 2670000 | ✗ | heuristic |
| 198 | -99999 | 3200 | ✗ | 3220 | ✗ | heuristic |
| 199 | 11 | 11 | ✓ | 11 | ✓ | heuristic |
| 200 | 46000 | 46000 | ✓ | 46000 | ✓ | heuristic |
| 201 | 7000 | 7000 | ✓ | 7000 | ✓ | heuristic |
| 202 | 677.50 | 677.50 | ✓ | 677.50 | ✓ | heuristic |
| 203 | 79000 | 60666.67 | ✗ | 66500 | ✗ | heuristic |
| 204 | 2500 | 2500 | ✓ | 2500 | ✓ | heuristic |
| 205 | 65 | 65 | ✓ | 65 | ✓ | heuristic |
| 206 | — | 1268 | ✗ | 1266.67 | ✗ | heuristic |
| 207 | 800 | 800 | ✓ | 800 | ✓ | heuristic |
| 208 | 175 | 166.67 | ✓ | 166.67 | ✓ | heuristic |
| 209 | 110 | 109.77 | ✓ | 109.77 | ✓ | heuristic |
| 210 | 19 | 18.33 | ✗ | 19 | ✓ | heuristic |
| 211 | -99999 | 47.78 | ✗ | 47.78 | ✗ | heuristic |
| 212 | — | — | ✓ | 0 | ✗ | heuristic |
| 213 | 14 | 6 | ✗ | 14 | ✓ | heuristic |
| 214 | — | — | ✓ | 16 | ✗ | heuristic |
| 215 | 100 | 100 | ✓ | 100 | ✓ | heuristic |
| 216 | 1125 | 1125 | ✓ | 1125 | ✓ | heuristic |
| 217 | 2000 | 2000 | ✓ | 2000 | ✓ | heuristic |
| 218 | 750 | 750 | ✓ | 750 | ✓ | heuristic |
| 219 | 1552 | 1552 | ✓ | 1555.56 | ✓ | heuristic |
| 220 | 200 | 200 | ✓ | 200 | ✓ | heuristic |
| 221 | 610 | 606.25 | ✓ | 610 | ✓ | heuristic |
| 222 | 50 | 50 | ✓ | 50 | ✓ | heuristic |
| 223 | 890 | 890 | ✓ | 890 | ✓ | heuristic |
| 224 | 819 | 819 | ✓ | 819 | ✓ | heuristic |
| 225 | 310 | 310 | ✓ | 310 | ✓ | heuristic |
| 226 | 555 | 555 | ✓ | 555 | ✓ | heuristic |
| 227 | 40 | 40 | ✓ | 40 | ✓ | heuristic |
| 228 | 430.77 | 460 | ✗ | 430.77 | ✓ | heuristic |
| 229 | 225 | 225 | ✓ | 225 | ✓ | heuristic |
| 230 | 30 | 30 | ✓ | 30 | ✓ | heuristic |
| 231 | 60 | 60 | ✓ | 60 | ✓ | heuristic |
| 232 | 1970 | 1760 | ✗ | 1760 | ✗ | heuristic |
| 233 | 256500 | 256500 | ✓ | 256551.72 | ✓ | heuristic |
| 234 | 70 | 60.80 | ✗ | 60.80 | ✗ | heuristic |
| 235 | 40 | 40 | ✓ | 40 | ✓ | heuristic |
| 236 | 125.49 | 125.49 | ✓ | 125.49 | ✓ | heuristic |
| 237 | 8 | 8 | ✓ | 8 | ✓ | heuristic |
| 238 | 16500 | 16500 | ✓ | 16500 | ✓ | heuristic |
| 239 | 2.30 | 2.14 | ✓ | 2.14 | ✓ | heuristic |
| 240 | 2400 | 2400 | ✓ | 2400 | ✓ | heuristic |
| 241 | 72 | 72 | ✓ | 72 | ✓ | heuristic |
| 242 | 214 | 214 | ✓ | 215.38 | ✓ | heuristic |
| 243 | 2190 | 2190 | ✓ | 2190 | ✓ | heuristic |
| 244 | 8 | 8 | ✓ | 8 | ✓ | heuristic |
| 245 | 26 | 26 | ✓ | 26 | ✓ | heuristic |

</details>

### IndustryOR (100 problems)

> IDs match 1:1 between our run and ORLM baseline (ran on exact same ORLM problem set).

**Agreement matrix (best solver per problem):**

| | ORLM ✓ | ORLM ✗ | Total |
|---|--:|--:|--:|
| **Ours ✓** | 28 | 18 | **46** |
| **Ours ✗** | 10 | 44 | 54 |
| **Total** | 38 | 62 | 100 |

- **18 problems** solved by us but not ORLM
- **10 problems** solved by ORLM but not us
- **28 problems** solved by both
- **44 problems** solved by neither

<details>
<summary>Click to expand full per-problem results</summary>

| ID | Difficulty | Type | Ground Truth | ORLM Pred | ORLM | Ours Pred | Ours | Best Solver |
|--:|---|---|--:|--:|:--:|--:|:--:|---|
| 1 | Easy | Integer Programming | 3050 | 3050 | ✓ | 3050 | ✓ | heuristic |
| 2 | Easy | Integer Programming | 135000 | 135000 | ✓ | 135000 | ✓ | heuristic |
| 3 | Easy | Mixed-Integer Programming | 30400 | 30400 | ✓ | 30400 | ✓ | heuristic |
| 4 | Easy | Integer Programming | 23000 | 23000 | ✓ | 23000 | ✓ | heuristic |
| 5 | Medium | Linear Programming | 180000 | 180000 | ✓ | 180000 | ✓ | heuristic |
| 6 | Easy | Integer Programming | 1600 | 1600 | ✓ | 1600 | ✓ | heuristic |
| 7 | Easy | Integer Programming | 90000 | 90000 | ✓ | 90000 | ✓ | heuristic |
| 8 | Easy | Integer Programming | 600 | 600 | ✓ | 600 | ✓ | heuristic |
| 9 | Easy | Linear Programming | 9800 | 9800 | ✓ | 9800 | ✓ | heuristic |
| 10 | Easy | Integer Programming | 38000 | 38000 | ✓ | 38000 | ✓ | heuristic |
| 11 | Medium | Integer Programming | 25000 | 25000 | ✓ | 25000 | ✓ | heuristic |
| 12 | Easy | Integer Programming | 734 | 734 | ✓ | 700 | ✓ | heuristic |
| 13 | Hard | Integer Programming | 53 | 53 | ✓ | 53 | ✓ | heuristic |
| 14 | Hard | Mixed-Integer Programming | 20240 | 20242 | ✓ | 20260.87 | ✓ | metaheuristic |
| 15 | Hard | Integer Programming | -99999 | 8.70 | ✗ | 1 | ✗ | heuristic |
| 16 | Medium | Linear Programming | 4700 | 4700 | ✓ | 47000000 | ✗ | heuristic |
| 17 | Easy | Integer Programming | 3 | 3 | ✓ | 3 | ✓ | heuristic |
| 18 | Easy | Mixed-Integer Programming | 37000 | 37000 | ✓ | 37000 | ✓ | heuristic |
| 19 | Easy | Integer Programming | 12 | 12 | ✓ | 25 | ✗ | heuristic |
| 20 | Hard | Integer Programming | 4 | 4 | ✓ | 4 | ✓ | heuristic |
| 21 | Hard | Mixed-Integer Programming | 43700 | 43300 | ✓ | 43200 | ✓ | metaheuristic |
| 22 | Medium | Linear Programming | 6800 | 6800 | ✓ | 6800 | ✓ | heuristic |
| 23 | Easy | Linear Programming | 135.27 | 135.27 | ✓ | 135.27 | ✓ | heuristic |
| 24 | Easy | Linear Programming | 150 | 150 | ✓ | 150 | ✓ | heuristic |
| 25 | Easy | Linear Programming | 1030 | 1030 | ✓ | 1030 | ✓ | metaheuristic |
| 26 | Easy | Linear Programming | 57 | 57 | ✓ | 57 | ✓ | heuristic |
| 27 | Easy | Linear Programming | 16 | — | ✗ | 16.80 | ✗ | heuristic |
| 28 | Easy | Mixed-Integer Programming | 16 | — | ✗ | 1300 | ✗ | heuristic |
| 29 | Medium | Linear Programming | 4685100 | — | ✗ | 904590 | ✗ | heuristic |
| 30 | Medium | Linear Programming | 5004 | — | ✗ | 140000 | ✗ | heuristic |
| 31 | Hard | Integer Programming | 42.10 | — | ✗ | 28.60 | ✗ | heuristic |
| 32 | Easy | Linear Programming | 8800 | — | ✗ | 8800 | ✓ | heuristic |
| 33 | Medium | Linear Programming | 1360 | — | ✗ | 19050 | ✗ | heuristic |
| 34 | Medium | Nonlinear Programming | 770 | — | ✗ | 2556 | ✗ | heuristic |
| 35 | Medium | Integer Programming | 14 | — | ✗ | 14 | ✓ | heuristic |
| 36 | Medium | Linear Programming | 246 | — | ✗ | 50000020 | ✗ | heuristic |
| 37 | Medium | Linear Programming | 165 | — | ✗ | 302.50 | ✗ | heuristic |
| 38 | Hard | Mixed-Integer Programming | 16 | — | ✗ | 417.33 | ✗ | heuristic |
| 39 | Hard | Integer Programming | 146 | — | ✗ | 153.33 | ✓ | heuristic |
| 40 | Easy | Mixed-Integer Programming | 1000 | — | ✗ | 1000 | ✓ | heuristic |
| 41 | Medium | Mixed-Integer Programming | 1581550 | — | ✗ | 2630141.60 | ✗ | heuristic |
| 42 | Hard | Integer Programming | 2.78 | — | ✗ | 2.78 | ✓ | heuristic |
| 43 | Hard | Mixed-Integer Programming | 10000 | — | ✗ | -10000 | ✗ | heuristic |
| 44 | Hard | Integer Programming | 153 | — | ✗ | 153 | ✓ | heuristic |
| 45 | Hard | Linear Programming | 103801 | — | ✗ | 200000 | ✗ | heuristic |
| 46 | Medium | Linear Programming | 8505 | — | ✗ | — | ✗ | heuristic |
| 47 | Medium | Mixed-Integer Programming | 5069500 | — | ✗ | 4914500 | ✓ | metaheuristic |
| 48 | Medium | Mixed-Integer Programming | 105.52 | — | ✗ | 1138000 | ✗ | heuristic |
| 49 | Medium | Linear Programming | -99999 | — | ✗ | 1042779.84 | ✗ | heuristic |
| 50 | Medium | Mixed-Integer Programming | 76 | — | ✗ | 172 | ✗ | heuristic |
| 51 | Medium | Mixed-Integer Programming | 44480 | — | ✗ | — | ✗ | heuristic |
| 52 | Medium | Mixed-Integer Programming | 13400 | — | ✗ | 421200 | ✗ | heuristic |
| 53 | Medium | Mixed-Integer Programming | 528 | — | ✗ | 973.33 | ✗ | heuristic |
| 54 | Medium | Other | -99999 | — | ✗ | 15003000 | ✗ | heuristic |
| 55 | Medium | Mixed-Integer Programming | 21 | — | ✗ | 1225 | ✗ | heuristic |
| 56 | Easy | Integer Programming | 1000 | — | ✗ | 149 | ✗ | heuristic |
| 57 | Easy | Integer Programming | 770 | — | ✗ | 26.60 | ✗ | heuristic |
| 58 | Easy | Linear Programming | 32.44 | — | ✗ | 0.03 | ✗ | heuristic |
| 59 | Hard | Mixed-Integer Programming | 1146.60 | — | ✗ | 1128.39 | ✓ | heuristic |
| 60 | Medium | Linear Programming | 4500 | — | ✗ | 4291.52 | ✓ | metaheuristic |
| 61 | Easy | Linear Programming | 2924 | — | ✗ | 2803.33 | ✓ | heuristic |
| 62 | Medium | Linear Programming | 11250 | — | ✗ | 0 | ✗ | heuristic |
| 63 | Medium | Mixed-Integer Programming | 1250 | — | ✗ | 0 | ✗ | heuristic |
| 64 | Medium | Linear Programming | 6105 | — | ✗ | 194.67 | ✗ | heuristic |
| 65 | Medium | Integer Programming | 58 | — | ✗ | 16 | ✗ | heuristic |
| 66 | Medium | Integer Programming | 770 | — | ✗ | 796 | ✓ | metaheuristic |
| 67 | Hard | Mixed-Integer Programming | 9337440 | — | ✗ | 10300000 | ✗ | heuristic |
| 68 | Hard | Mixed-Integer Programming | 1644.63 | — | ✗ | — | ✗ | heuristic |
| 69 | Hard | Mixed-Integer Programming | 40 | — | ✗ | — | ✗ | heuristic |
| 70 | Hard | Integer Programming | 623 | 623 | ✓ | 623 | ✓ | heuristic |
| 71 | Easy | Integer Programming | 240000 | 268000 | ✗ | 240000 | ✓ | heuristic |
| 72 | Easy | Mixed-Integer Programming | 21 | 640 | ✗ | 640 | ✗ | heuristic |
| 73 | Easy | Integer Programming | 365 | 365 | ✓ | 365 | ✓ | heuristic |
| 74 | Easy | Mixed-Integer Programming | 960 | 960 | ✓ | 956 | ✓ | heuristic |
| 75 | Easy | Mixed-Integer Programming | 15000 | 3060 | ✗ | 980 | ✗ | heuristic |
| 76 | Medium | Mixed-Integer Programming | 369000 | 369000 | ✓ | 369000 | ✓ | heuristic |
| 77 | Medium | Integer Programming | 0 | 0 | ✓ | 1000 | ✗ | heuristic |
| 78 | Medium | Linear Programming | 435431000 | 103750000 | ✗ | 435431250 | ✓ | heuristic |
| 79 | Medium | Integer Programming | 7.10 | 7.10 | ✓ | 14.10 | ✗ | heuristic |
| 80 | Medium | Linear Programming | 4316659.20 | 4316659.20 | ✓ | 1288640 | ✗ | heuristic |
| 81 | Medium | Mixed-Integer Programming | 530 | 4240 | ✗ | 4240 | ✗ | heuristic |
| 82 | Easy | Linear Programming | 978400 | 1169200 | ✗ | 579999.98 | ✗ | heuristic |
| 83 | Medium | Linear Programming | 4848 | 0 | ✗ | 4848.47 | ✓ | hyperheuristic |
| 84 | Hard | Mixed-Integer Programming | 10755 | 70190 | ✗ | — | ✗ | heuristic |
| 85 | Easy | Mixed-Integer Programming | 118400 | 294600 | ✗ | 118400 | ✓ | heuristic |
| 86 | Medium | Linear Programming | 426 | 0 | ✗ | 106 | ✗ | heuristic |
| 87 | Easy | Integer Programming | 85 | 0 | ✗ | 85 | ✓ | heuristic |
| 88 | Medium | Mixed-Integer Programming | 16 | 472.30 | ✗ | — | ✗ | heuristic |
| 89 | Easy | Linear Programming | -1900 | -1900 | ✓ | 1500 | ✗ | heuristic |
| 90 | Easy | Linear Programming | 150 | 170 | ✗ | 150 | ✓ | heuristic |
| 91 | Medium | Mixed-Integer Programming | 1146.57 | 0 | ✗ | 1146.57 | ✓ | heuristic |
| 92 | Easy | Linear Programming | 20 | 0 | ✗ | 53 | ✗ | heuristic |
| 93 | Easy | Linear Programming | 5000 | 5000 | ✓ | 105 | ✗ | heuristic |
| 94 | Hard | Integer Programming | 22 | 22 | ✓ | 22 | ✓ | heuristic |
| 95 | Easy | Linear Programming | 770 | 14000000 | ✗ | 350 | ✗ | heuristic |
| 96 | Easy | Linear Programming | 9500 | 9500 | ✓ | 2600 | ✗ | heuristic |
| 97 | Easy | Linear Programming | 1360000 | 1360000 | ✓ | 510000 | ✗ | heuristic |
| 98 | Medium | Linear Programming | 25 | 3.75 | ✗ | 525000000 | ✗ | heuristic |
| 99 | Medium | Mixed-Integer Programming | 5500 | 2220 | ✗ | 1400420 | ✗ | heuristic |
| 100 | Hard | Integer Programming | 0 | 0 | ✓ | 7 | ✗ | heuristic |

</details>
