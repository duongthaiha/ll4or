# Large Language Models in Operations Research: Automatic Solution Generation

**Since 2022, a rapidly expanding body of academic and applied research has demonstrated that LLMs can automate core operations research tasks — from translating natural-language problem descriptions into formal mathematical models to generating executable solver code and, in some cases, directly proposing solutions.** The most effective systems combine LLMs with classical OR solvers to guarantee feasibility and optimality, while pure LLM approaches show promise on smaller-scale problems but face fundamental scalability constraints.

[Image omitted]

[^1] [^2] [^3] [^4]

---

## 1. LLMs for Automated Optimisation Modelling and Solving

### NL4Opt: The Foundational Competition (NeurIPS 2022)

The **Natural Language for Optimization (NL4Opt) Competition** at NeurIPS 2022 was the first community effort to benchmark LLM-based extraction and formulation of optimisation problems from text. The competition defined two sub-tasks: (1) recognising and labelling semantic entities corresponding to optimisation problem components and (2) generating a meaning representation (logical form) of the problem. The first-place team (**Infrrd AI Lab**, members: JiangLong He, Mamatha N., Shiv Vignesh, Deepak Kumar, Akshay Uppal) used **ensemble learning with text augmentation and segment shuffling**, achieving a **test F1-score of 0.939** on entity tagging. The second-place team (**mcmc**: Kangxu Wang, Ze Chen, Jiewen Zheng, from OPD) also competed on this sub-task. While NL4Opt focused on problem formulation rather than solution generation, it established the first public OR-modelling benchmark and catalysed subsequent research into end-to-end automation.

### OR-LLM-Agent: End-to-End Automated Solving (2025)

**OR-LLM-Agent** (Bowen Zhang and Pengcheng Luo, arXiv:2503.10009v2) represents the state of the art in fully automated OR problem solving. The system pipeline works in three sequential stages[^5] [^6]:

1. A **Math Agent** (using a reasoning LLM with chain-of-thought) translates a natural-language problem description into a formal mathematical model[^7].
2. A **Code Agent** generates executable Python code targeting the **Gurobi** solver[^8].
3. A **Debugging Agent** executes the code, catches runtime errors, and performs iterative self-repair (code-level fixes) or self-verification (model-level corrections when no feasible solution exists)[^9].

The authors also constructed **BWOR**, a high-quality benchmark for evaluating LLMs on OR tasks. Their analysis revealed that existing benchmarks such as NL4OPT "may contain a non-trivial number of incorrect answers, potentially undermining the reliability of model evaluation"[^10]. In contrast, BWOR provides more discriminative assessment. On BWOR, **reasoning LLMs consistently outperformed their non-reasoning counterparts**, with accuracy improvements ranging from **10.98% to 35.37%** across model families. This stood in contrast to older benchmarks where reasoning LLMs sometimes *underperformed* — for example, on IndustryOR, GPT-o4-mini achieved **5.00% lower accuracy** than GPT-4o, and on MAMO-Complex, Gemini 2.5 Pro underperformed Gemini 2.0 Flash by **7.11%** — suggesting that the benchmark quality, not the model, was the bottleneck.

Across five OR datasets, OR-LLM-Agent outperformed advanced methods including **GPT-o3, Gemini 2.5 Pro, and ORLM by at least 7%** in accuracy[^11]. Ablation studies revealed the value of decomposition: the Math Agent + Code Agent configuration improved average accuracy by **4.07%** over Code Agent alone, and adding the Debugging Agent to form the full pipeline further improved accuracy by **5.49%**. The Debugging Agent also reduced the average code error rate by **3.74%** across all datasets. Code and datasets have been publicly released on GitHub and HuggingFace[^12].

**Tradeoff:** OR-LLM-Agent's reliance on a commercial solver (Gurobi) means mathematical rigour in the final solution is guaranteed by the solver, not the LLM. The LLM's contribution lies in modelling and code generation — steps where errors can propagate despite self-repair. Additionally, the system requires no fine-tuning or retraining[^13], making it more accessible but potentially less adaptable to highly specialised OR sub-domains.

### Chain-of-Experts: Multi-Agent LLM Framework (ICLR 2024)

**Chain-of-Experts (CoE)** (Xiao, Zhang, Wu et al., from Zhejiang University and Huawei Noah's Ark Lab) was published at **ICLR 2024**. Rather than using a single LLM, CoE deploys multiple specialised LLM-based agents — each assigned a specific role and endowed with domain knowledge related to OR — coordinated by a "Conductor" agent. The approach "employs cooperative agents with OR knowledge to solve linear programming problems, achieving strong results on standard and complex benchmarks"[^14]. In the accuracy comparison table within OR-LLM-Agent's evaluation, Chain-of-Experts achieved **64.20%** on one standard benchmark (likely NL4OPT), compared to **47.90%** for tag-BART and **78.80%** for OptiMUS.

**Tradeoff:** CoE requires carefully designed role-specific prompts and domain knowledge bases for each agent, which limits out-of-the-box applicability to new OR domains. Its multi-agent design adds orchestration complexity. Nonetheless, the framework demonstrates that **decomposing OR modelling across specialised LLM agents can substantially improve formulation accuracy** over single-model prompting.

### ORQA: Evaluating Generalisation (AAAI 2025)

**ORQA** (Mostajabdaveh, Yu, Dash, Ramamonjison, Byusa, Carenini, Zhou & Zhang, from Huawei Technologies Canada, University of Toronto, and University of British Columbia) is a benchmark published in the *Proceedings of the AAAI Conference on Artificial Intelligence*, Vol. 39 No. 23, pages 24902–24910 (2025)[^15]. It assesses whether LLMs can emulate the knowledge and reasoning skills of OR experts when given diverse and complex optimisation problems that require **multistep reasoning** to build mathematical models[^16]. Evaluations of open-source LLMs including **LLaMA 3.1, DeepSeek, and Mixtral** revealed **"modest performance, indicating a gap in their aptitude to generalize to specialized technical domains"**[^17]. This underscores that while LLMs have made progress on curated benchmarks, generalisation to novel OR problem formulations remains an open challenge.

### The Benchmarking Landscape

Within three years, a rich set of OR-LLM benchmarks has emerged. The OR-LLM-Agent paper evaluates across five datasets: **BWOR** (their own high-quality benchmark), **NL4OPT**, **MAMO** (with Easy, Complex, and ODE categories), and **IndustryOR**[^18]. Additional notable projects include **OptiMUS** (AhmadiTeshnizi, Gao & Udell, 2024), which uses a modular agent for (MI)LP modelling with "over 20%–30% accuracy gains"[^19], and **ORLM** (Huang et al., 2025), which trains open-source LLMs on synthetic data via a customisable pipeline[^20]. The survey project *"A Survey of Optimization Modeling Meets LLMs: Progress and Future Directions"* catalogues this rapidly expanding landscape.

The progression from simple LP word problems (NL4Opt, 2022) to complex industrial OR benchmarks (BWOR, 2025) illustrates the field's growing maturity. The finding that existing benchmarks like NL4OPT contained errors that masked model capabilities[^21] underscores that **benchmark quality, not just size, is critical** for reliably evaluating LLM performance on OR tasks.

---

## 2. LLMs as Direct Problem Solvers

Beyond formulating models, researchers are investigating LLMs as the solution engine itself — using generative and reasoning abilities to directly search for or propose solutions.

### Optimisation by Prompting (OPRO)

**OPRO** (Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou & Xinyun Chen; **ICLR 2024**) proposes a simple and effective approach to leverage LLMs as optimisers, where the optimisation task is described in natural language[^22]. In each step, the LLM generates new solutions from a prompt containing previously generated solutions with their evaluation values; these are then evaluated and added to the prompt for the next iteration[^23].

Yang et al. first demonstrated OPRO on **linear regression and traveling salesman problems (TSP)**, then applied it to prompt optimisation — finding instructions that maximise task accuracy[^24]. Results showed that the best prompts optimised by OPRO outperformed human-designed prompts by **up to 8% on GSM8K** and by **up to 50% on Big-Bench Hard tasks**[^25]. The paper spans 42 pages with 26 figures and 15 tables[^26].

Related work from the scheduling literature confirms that "in small-scale optimization scenarios, LLMs can generate high-quality solutions solely through prompting, sometimes **matching or even surpassing the performance of manually crafted heuristic algorithms**"[^27].

**Tradeoff:** OPRO requires no additional training and minimal domain knowledge — the LLM's own understanding guides the search. However, performance depends on the LLM's reasoning capacity and the search space size. The approach is best suited for problems where evaluation is cheap but design is complex (like prompt engineering), and struggles to scale to large combinatorial instances where the solution space overwhelms the LLM's reasoning window.

### LLM-Driven Evolutionary Algorithm (LMEA)

**LMEA** (Shengcai Liu, Caishun Chen, Xinghua Qu, Ke Tang & Yew-Soon Ong; **accepted at CEC 2024**) presents the first study on LLMs as evolutionary combinatorial optimisers[^28]. In each generation, LMEA instructs the LLM to **select parent solutions** from the current population and **perform crossover and mutation** to generate offspring solutions, which are then evaluated and included in the population for the next generation[^29]. LMEA is equipped with a **self-adaptation mechanism** that controls the LLM's temperature, enabling it to balance exploration and exploitation and prevent stagnation in local optima[^30].

The main advantage is that this approach **requires minimal domain knowledge and human effort, and no additional model training**[^31]. On classical TSP instances, the results showed that LMEA **"performs competitively to traditional heuristics in finding high-quality solutions on TSP instances with up to 20 nodes"**[^32].

