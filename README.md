# Logistic-AI: SE-RAG for Logistics Anomaly Dispatch

Structured Experience-Augmented Generation (SE-RAG) for anomaly dispatch in logistics — a RAG-based decision framework that combines rule mining, causal reasoning, and LLM inference.

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b)](https://arxiv.org/abs/xxxx.xxxxx)

---

## The Problem

**Logistics anomaly dispatch is a multi-solution problem, not a classification problem.**

Given the same anomaly (vehicle breakdown, traffic jam, order cancellation), multiple dispatch actions may all be reasonable — `delay_tolerant`, `reroute`, `reassign_order` are all viable depending on dispatcher preference, fleet state, and implicit constraints. **~48% of scenarios have multiple valid dispatch actions** in our dataset. This is not data noise — it is an inherent characteristic of the domain.

### Why Existing Methods Fall Short

| Method | Limitation |
|---|---|
| Classifier | One-to-one hard label mapping, fundamentally ill-suited for multi-solution spaces |
| Plain LLM | Lacks domain experience; accuracy far below acceptable |
| Standard RAG (BM25/TF-IDF/Dense) | Retrieves scattered individual cases without structured guidance; marginal gains |
| **SE-RAG (ours)** | Rules = distilled experience patterns; LLM decides within rule + causal evidence while preserving multi-solution space |

---

## SE-RAG Architecture

![SE-RAG Architecture](architecture.png)

### Offline Stage: Knowledge Construction

1. **Rule Mining** — Mine event-feature → action-decision association rules from 30k historical cases under a unified feature representation space, producing ~44k raw rules
2. **Causal Graph Extraction** — Extract causal strength between features and action choices, helping the LLM understand *why* a feature influences an action
3. **Weighted Pruning** — Weight minority actions (reroute, reassign_order, adjust_capacity) to mitigate class imbalance; retain ~17k high-quality rules (conf ≥ 0.7, sup ≥ 5)

### Online Stage: Retrieval-Augmented Decision

1. Input anomaly event features (event type, severity, load rate, urgent order count, etc., 24 dimensions)
2. Rule matching + causal reasoning signal retrieval
3. Construct prompt context (rule evidence + causal graph + feature-similar cases)
4. LLM outputs final dispatch decision grounded in structured evidence

---

## Results

### Main Results (1000 queries, deepseek-v4-flash)

| Method | Multi-GT | Single (Exact) | Avg Latency |
|---|---|---|---|
| **SE-RAG** | **90.4%** | **63.3%** | **7.09s** |
| bm25_rag | 86.9% | 54.9% | 8.80s |
| dense_rag | 76.6% | 30.4% | 8.81s |
| tfidf_rag | 80.1% | 45.3% | 8.86s |
| no_rag (plain LLM) | 69.8% | 23.5% | 7.32s |

### Per-Action Performance (SE-RAG vs No-RAG, 1000 queries, Multi-GT)

| Action | Proportion | No-RAG | SE-RAG | Delta |
|---|---|---|---|---|
| delay_tolerant | 48.2% | 62.6% | **97.4%** | **+34.8pt** |
| ignore | 2.8% | 65.2% | **78.3%** | **+13.0pt** |
| adjust_capacity | 15.2% | 68.6% | **78.0%** | **+9.4pt** |
| reassign_order | 22.8% | 78.7% | **85.6%** | **+6.9pt** |
| reroute | 11.0% | 84.8% | **90.2%** | **+5.4pt** |

---

## Repository Structure

```text
logistic-ai/
├── architecture.png              # SE-RAG architecture diagram
├── run.py                        # Unified entry point (5 subcommands)
├── config/
│   ├── llm.local.example.json    # LLM config template
│   └── llm.local.json            # Local API key (not committed)
├── data/
│   ├── cases/                    # Historical cases (gitignored)
│   └── rules/                    # Rule knowledge base
│       ├── rules.jsonl           # ~44k raw rules
│       ├── pruned/               # Pruned rule sets
│       └── causal_graph.json     # Causal graph
├── docs/                         # Architecture & design docs
├── scripts/                      # Core scripts
│   ├── parallel_case_generator.py
│   ├── generate_validation_queries.py
│   ├── eval_se_rag.py
│   └── eval_standard_baselines.py
├── src/
│   ├── agent/                    # LLM gateway & decision module
│   ├── optimizer/                # OR-Tools VRP solver
│   ├── rag/                      # Retrieval, rules, experience model
│   └── simulation/               # Simulation environment
├── record/                       # Experiment logs / paper notes
├── output/                       # Experiment output (gitignored)
└── README.md
```

---

## Setup

Python 3.10+, using conda:

```powershell
conda create -n logistic python=3.10
conda activate logistic
pip install ortools fastapi uvicorn streamlit numpy scikit-learn pydantic
```

---

## LLM Configuration

Edit `config/llm.local.json`:

```json
{
  "provider": "openai",
  "api_key": "your-api-key-here",
  "model": "deepseek-v4-flash",
  "base_url": "https://qianweikeji.fun/v1",
  "timeout": 120
}
```

---

## Usage

All operations go through `run.py`:

```powershell
# Generate 30k simulated cases
python run.py generate-cases --count 30000

# Sample 1000 validation queries (distribution-preserving)
python run.py generate-validation --count 1000

# Run SE-RAG vs No-RAG comparison (default 200 queries)
python run.py evaluate-se-rag --query-count 200 --seed 20260428

# Full evaluation (1000 queries)
python run.py evaluate-se-rag --query-count 1000 --seed 20260428

# Run all standard baselines
python run.py evaluate-baselines --query-count 1000 --seed 20260428 --methods all

# Single baseline run (for parallel execution)
python run.py evaluate-baselines --query-count 1000 --method no_rag
python run.py evaluate-baselines --query-count 1000 --method bm25_rag

# See full parameters
python run.py evaluate-se-rag --help
```

> Internally delegates to the corresponding `scripts/` module.

---

## Git Boundary

**Commit:**
- `src/` — business logic
- `scripts/*.py` — experiment scripts
- `config/*.example.json`
- `docs/`
- `README.md`
- `architecture.png`

**Do not commit:**
- `record/`, `output/`
- `data/cases/`, `data/rules/pruned/`
- `*.pkl`, `*.pyc`, `__pycache__/`
- `config/llm.local.json`
- `paper/`, `notebooks/paper/`
- `scripts/generate_paper.py`

---

## Design Principles

1. **Solver handles constraints** — OR-Tools manages vehicle routing, capacity, and time windows
2. **RAG provides experiential evidence** — rules + causal graph as decision context
3. **LLM preserves multi-solution space** — no forced single answer; LLM chooses within plausible alternatives
4. **Controlled LLM output** — fixed action set + JSON schema reduces risk
5. **Artifacts stay local** — experiment outputs and paper drafts never enter the repo

---

## Citation

```bibtex
@misc{se-rag-logistics,
  author = {Ken},
  title = {SE-RAG: Structured Experience-Augmented Generation for Logistics Anomaly Dispatch},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/xkiller747x/Experience-Based-Flow}
}
```
