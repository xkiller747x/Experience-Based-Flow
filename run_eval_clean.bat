@echo off
cd /d D:\code\logistic-ai
if exist output\se_rag_eval\checkpoint.jsonl del output\se_rag_eval\checkpoint.jsonl
if exist output\se_rag_eval\run.log del output\se_rag_eval\run.log
echo Starting at %TIME% > output\se_rag_eval\run.log
D:\Conda\mini\envs\logistic\python.exe -u scripts/eval_se_rag.py --query-count 200 --seed 20260428 --output-dir output/se_rag_eval >> output\se_rag_eval\run.log 2>&1
echo Done at %TIME% >> output\se_rag_eval\run.log
