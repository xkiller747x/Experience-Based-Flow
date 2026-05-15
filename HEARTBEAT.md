# HEARTBEAT.md - Overnight Logistics AI Experiments

## Goal
Improve logistics action prediction accuracy beyond current best (positive_only: 63.0%).

## Root Cause
Historical case labels ≠ solver GT → retrieved cases bias LLM toward delay_tolerant.

## Running Experiments (parallel sub-agents)

### Exp 1: Solver-relabeled retrieval pool
- Relabel 500-1000 cases with solver GT
- Test bm25_rag / positive_only with relabeled pool
- Status: [ ] pending

### Exp 2: Prompt optimization on positive_only  
- Try top-10, action-balanced sampling, explicit bias warnings
- Status: [ ] pending

### Exp 3: Direct ML classifier (solver mimic)
- Train LightGBM on solver-GT queries → predict action from features
- No LLM needed, pure ML
- Status: [ ] pending

## Results to report
- [ ] Baseline: positive_only 63.0%
- [ ] Exp 1 result
- [ ] Exp 2 result  
- [ ] Exp 3 result
