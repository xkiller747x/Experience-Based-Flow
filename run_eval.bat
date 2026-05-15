@echo off
cd /d D:\code\logistic-ai
D:\Conda\mini\envs\logistic\python.exe -u scripts/eval_se_rag.py --query-count 200 --seed 20260428 --output-dir output\se_rag_eval > output\se_rag_eval\run.log 2>&1
