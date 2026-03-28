# OR-LLM Benchmark Datasets

A catalogue of publicly available benchmarks for evaluating Large Language Models on Operations Research tasks — identified from the literature surveyed in [`research.md`](research.md).

---

## Quick-Reference Table

| Benchmark | Venue / Year | Problem Types | Size | License | Download |
|-----------|-------------|---------------|------|---------|----------|
| [NL4Opt](#1-nl4opt) | NeurIPS 2022 | LP entity tagging & formulation | ~1,100 LP problems | MIT | [GitHub](https://github.com/nl4opt/nl4opt-competition) · [HuggingFace](https://huggingface.co/datasets/CardinalOperations/NL4OPT) |
| [BWOR](#2-bwor) | arXiv 2025 (SJTU) | Mixed real-world OR (LP, MIP, NLP, …) | ~83 problems (EN + ZH) | — | [HuggingFace](https://huggingface.co/datasets/SJTU/BWOR) · [GitHub](https://github.com/bwz96sco/or_llm_agent) |
| [MAMO](#3-mamo) | arXiv 2024 | LP (Easy & Complex), ODE | Multiple splits (JSONL) | CC BY-NC 4.0 | [GitHub](https://github.com/FreedomIntelligence/Mamo) · [HuggingFace](https://huggingface.co/datasets/CardinalOperations/MAMO) |
| [IndustryOR](#4-industryor) | Ops Research 2025 | LP, IP, MIP, NLP | 100 real-world problems | CC BY-NC 4.0 | [HuggingFace](https://huggingface.co/datasets/CardinalOperations/IndustryOR) · [GitHub](https://github.com/Cardinal-Operations/ORLM) |
| [ORQA](#5-orqa) | AAAI 2025 | OR multiple-choice QA | 1,468 test + 45 val | — | [GitHub](https://github.com/nl4opt/ORQA) |
| [OptiMUS / NLP4LP](#6-optimus--nlp4lp) | ICML 2024 | LP, MILP | Expert-annotated NL→model pairs | CC BY-NC 4.0 | [GitHub](https://github.com/teshnizi/OptiMUS) · [HuggingFace](https://huggingface.co/datasets/udell-lab/NLP4LP) |
| [ORLM / OR-Instruct](#7-orlm--or-instruct) | arXiv 2025 | LP, MIP, NIP (± tabular data) | Synthetic pipeline + 100 IndustryOR | — | [GitHub](https://github.com/Cardinal-Operations/ORLM) · [HuggingFace (model)](https://huggingface.co/CardinalOperations/ORLM-LLaMA-3-8B) |

---

## 1. NL4Opt

**Full name:** Natural Language for Optimization  
**Venue:** NeurIPS 2022 Competition Track  
**Paper:** [NL4Opt Competition: Formulating Optimization Problems Based on Their Natural Language Descriptions](https://proceedings.mlr.press/v220/ramamonjison23a.html)  
**Website:** <https://nl4opt.github.io/>

### Description

The first community-scale benchmark for converting natural-language optimisation problem descriptions into formal mathematical models. The competition defined two sub-tasks:

1. **Entity recognition** — label semantic entities (decision variables, constraints, objective) in problem text.
2. **Generation** — produce a meaning representation (logical form) of the LP problem.

The dataset contains ~1,100 linear-programming word problems with annotated entities and canonical formulations.

### Download

| Source | Link |
|--------|------|
| GitHub (official) | <https://github.com/nl4opt/nl4opt-competition> |
| HuggingFace (test set variant) | <https://huggingface.co/datasets/CardinalOperations/NL4OPT> |

```bash
# Clone the official repository
git clone https://github.com/nl4opt/nl4opt-competition.git
# Data lives in generation_data/ and ner_data/
```

```python
# Or load via HuggingFace
from datasets import load_dataset
ds = load_dataset("CardinalOperations/NL4OPT")
```

### Notes

- The HuggingFace version adds programmatic answer checking but is not the official ground-truth release.
- OR-LLM-Agent (2025) noted that NL4OPT "may contain a non-trivial number of incorrect answers" — keep this in mind when interpreting evaluation results.

---

## 2. BWOR

**Full name:** Benchmark for Writing Operations Research  
**Venue:** arXiv 2025 — Shanghai Jiao Tong University  
**Paper:** [OR-LLM-Agent: Automating Modeling and Solving of Operations Research Optimization Problems](https://arxiv.org/abs/2503.10009)

### Description

A high-quality, curated benchmark of ~83 real-world OR problems described in natural language (English and Chinese). Designed to be more discriminative than earlier benchmarks (NL4OPT, MAMO). On BWOR, reasoning LLMs consistently outperformed non-reasoning counterparts by 10.98–35.37 % accuracy.

### Download

| Source | Link |
|--------|------|
| HuggingFace | <https://huggingface.co/datasets/SJTU/BWOR> |
| GitHub (OR-LLM-Agent code) | <https://github.com/bwz96sco/or_llm_agent> |

```python
from datasets import load_dataset
ds = load_dataset("SJTU/BWOR")
```

### Notes

- Problems span LP, MIP, NLP, and other OR types.
- Code and evaluation scripts are in the GitHub repo.

---

## 3. MAMO

**Full name:** Mamo: a Mathematical Modeling Benchmark with Solvers  
**Venue:** arXiv 2024  
**Paper:** [Mamo: a Mathematical Modeling Benchmark with Solvers](https://arxiv.org/abs/2405.13144)

### Description

A benchmark for evaluating LLMs' mathematical modelling abilities on optimisation problems. Organised into multiple splits:

| Split | Description |
|-------|-------------|
| **Easy LP** | Straightforward linear-programming problems |
| **Complex LP** | Multi-constraint / larger LP problems |
| **ODE** | Ordinary differential equation modelling tasks |

Each data point is a JSONL record with an OR/optimisation question and its answer.

### Download

| Source | Link |
|--------|------|
| GitHub | <https://github.com/FreedomIntelligence/Mamo> |
| HuggingFace | <https://huggingface.co/datasets/CardinalOperations/MAMO> |

```bash
git clone https://github.com/FreedomIntelligence/Mamo.git
ls Mamo/Data/
# mamo_easy_lp.jsonl  mamo_complex_lp.jsonl  ...
```

```python
from datasets import load_dataset
ds = load_dataset("CardinalOperations/MAMO", "easy_lp")
```

**License:** CC BY-NC 4.0

---

## 4. IndustryOR

**Full name:** Industry Operations Research Benchmark  
**Venue:** Published alongside ORLM in *Operations Research* (2025)  
**Paper:** [ORLM: A Customizable Framework in Training Large Models for Automated Optimization Modeling](https://arxiv.org/abs/2405.17743)

### Description

100 real-world industrial optimisation problems spanning multiple problem types and three difficulty levels:

| Dimension | Categories |
|-----------|-----------|
| **Problem types** | LP, IP, MIP, NLP |
| **Difficulty** | Easy, Medium, Hard |

Each entry includes a natural-language description, solution(s), and metadata (type, difficulty).

### Download

| Source | Link |
|--------|------|
| HuggingFace | <https://huggingface.co/datasets/CardinalOperations/IndustryOR> |
| GitHub (part of ORLM) | <https://github.com/Cardinal-Operations/ORLM> |

```python
from datasets import load_dataset
ds = load_dataset("CardinalOperations/IndustryOR")
```

**License:** CC BY-NC 4.0

---

## 5. ORQA

**Full name:** Operations Research Question Answering  
**Venue:** AAAI 2025 (Proceedings, Vol. 39, No. 23, pp. 24902–24910)  
**Paper:** [Evaluating LLM Reasoning in the Operations Research Domain with ORQA](https://arxiv.org/abs/2412.17874)

### Description

A multiple-choice QA benchmark crafted by OR experts to test whether LLMs can emulate expert-level reasoning on diverse, complex optimisation problems requiring multi-step reasoning. Evaluations of LLaMA 3.1, DeepSeek, and Mixtral revealed "modest performance, indicating a gap in their aptitude to generalise to specialised technical domains."

| Split | Instances | Contents |
|-------|-----------|----------|
| **Test** | 1,468 | CONTEXT + QUESTION + 4 OPTIONS + TARGET_ANSWER |
| **Validation** | 45 | Same as test + step-by-step expert REASONING |

### Download

| Source | Link |
|--------|------|
| GitHub | <https://github.com/nl4opt/ORQA> |

```bash
git clone https://github.com/nl4opt/ORQA.git
# Dataset files: src/task/dataset/ORQA_test.jsonl, ORQA_validation.jsonl
```

---

## 6. OptiMUS / NLP4LP

**Full name:** Optimization Modeling Using MIP Solvers and Large Language Models  
**Venue:** ICML 2024  
**Paper:** [OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models](https://arxiv.org/abs/2402.10172)  
**Authors:** AhmadiTeshnizi, Gao & Udell

### Description

OptiMUS is a modular LLM-based agent for LP/MILP modelling that achieved "over 20–30 % accuracy gains" on standard benchmarks. The associated dataset, **NLP4LP**, contains expert-annotated natural-language optimisation problems mapped to formal LP/MILP models.

### Download

| Source | Link |
|--------|------|
| GitHub (code + data) | <https://github.com/teshnizi/OptiMUS> |
| HuggingFace (NLP4LP dataset) | <https://huggingface.co/datasets/udell-lab/NLP4LP> |

```bash
git clone https://github.com/teshnizi/OptiMUS.git
```

```python
from datasets import load_dataset
ds = load_dataset("udell-lab/NLP4LP")
```

**License:** CC BY-NC 4.0

---

## 7. ORLM / OR-Instruct

**Full name:** ORLM — Training Large Language Models for Optimization Modeling  
**Venue:** arXiv 2025, accepted in *Operations Research*  
**Paper:** [ORLM: A Customizable Framework in Training Large Models for Automated Optimization Modeling](https://arxiv.org/abs/2405.17743)  
**Authors:** Huang et al. (Cardinal Operations)

### Description

ORLM introduces **OR-Instruct**, a semi-automated pipeline to generate synthetic training data for OR modelling. The project also provides fine-tuned open-source LLMs (e.g., ORLM-LLaMA-3-8B) that achieve state-of-the-art results on NL4OPT, MAMO, and IndustryOR. The synthetic data covers LP, MIP, and NIP problems — with and without tabular data.

### Download

| Source | Link |
|--------|------|
| GitHub (code + sample data) | <https://github.com/Cardinal-Operations/ORLM> |
| HuggingFace (model) | <https://huggingface.co/CardinalOperations/ORLM-LLaMA-3-8B> |

```bash
git clone https://github.com/Cardinal-Operations/ORLM.git
cd ORLM
# Sample data in data/ directory; see README for full instructions
```

---

## Local Dataset Paths

All datasets have been downloaded into the `datasets/` directory:

```
datasets/
├── BWOR/           (199M) — OR-LLM-Agent repo + 5 benchmark JSONs
│   └── data/datasets/
│       ├── BWOR.json
│       ├── IndustryOR.json
│       ├── MAMO_ComplexLP.json
│       ├── MAMO_EasyLP.json
│       └── NL4OPT_with_optimal_solution.json
├── NL4Opt/         (18M)  — Official competition data
│   └── generation_data/
│       ├── train.jsonl
│       ├── dev.jsonl
│       └── test.jsonl
├── ORQA/           (11M)  — AAAI 2025 QA benchmark
│   └── dataset/
│       ├── ORQA_test.jsonl      (1,468 instances)
│       └── ORQA_validation.jsonl (45 instances)
├── IndustryOR/     (280K) — From HuggingFace
│   └── IndustryOR.json
├── MAMO/           (3.6M) — GitHub repo + HuggingFace data
│   └── HF_data/
│       ├── MAMO_EasyLP.json
│       └── MAMO_ComplexLP.json
├── OptiMUS/        (64M)  — Full codebase (RAG data included)
│   └── (NLP4LP dataset requires HuggingFace auth — see note below)
└── ORLM/           (15M)  — Code + sample data + eval results
```

> **Note:** The **NLP4LP** dataset (used by OptiMUS) is access-restricted on HuggingFace.
> To download it, authenticate first:
> ```bash
> pip install huggingface_hub
> huggingface-cli login
> # Then:
> from datasets import load_dataset
> ds = load_dataset("udell-lab/NLP4LP")
> ```

---

## Additional Resources

- **LLM4OR Survey Portal:** <https://llm4or.github.io/LLM4OR/> — comprehensive catalogue of optimisation-modelling + LLM research.
- **OR-LLM-Agent evaluation code:** <https://github.com/bwz96sco/or_llm_agent> — scripts to run evaluations across all five core benchmarks (BWOR, NL4OPT, MAMO-Easy, MAMO-Complex, IndustryOR).
