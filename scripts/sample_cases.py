import json, random
random.seed(20260428)
with open(r'D:\code\logistic-ai\data\cases\cases_v2_solver.jsonl', encoding='utf-8') as f:
    lines = f.readlines()
sample = random.sample(lines, 30000)
with open(r'D:\code\logistic-ai\data\cases\cases_30k.jsonl', 'w', encoding='utf-8') as f:
    f.writelines(sample)
print(f'Sampled {len(sample)} lines written')
