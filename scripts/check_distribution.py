import json
from collections import Counter

# Original 1M distribution
event_types = Counter()
scenarios = Counter()
actions = Counter()
total = 0

with open(r'D:\code\logistic-ai\data\cases\cases_v2_solver.jsonl', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        event_types[r['event_type']] += 1
        scenarios[r['scenario']] += 1
        actions[r['action']] += 1
        total += 1

print(f"=== Original ({total:,} cases) ===")
print("\nEvent types:")
for k, v in sorted(event_types.items()):
    print(f"  {k}: {v:>8,} ({v/total*100:.2f}%)")
print("\nScenarios:")
for k, v in sorted(scenarios.items()):
    print(f"  {k}: {v:>8,} ({v/total*100:.2f}%)")
print("\nActions:")
for k, v in sorted(actions.items()):
    print(f"  {k}: {v:>8,} ({v/total*100:.2f}%)")

# Sampled 30k distribution
event_types2 = Counter()
scenarios2 = Counter()
actions2 = Counter()
total2 = 0

with open(r'D:\code\logistic-ai\data\cases\cases_30k.jsonl', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        event_types2[r['event_type']] += 1
        scenarios2[r['scenario']] += 1
        actions2[r['action']] += 1
        total2 += 1

print(f"\n=== Sampled 30k ({total2:,} cases) ===")
print("\nEvent types:")
for k in sorted(event_types.keys()):
    v = event_types2.get(k, 0)
    print(f"  {k}: {v:>6,} ({v/total2*100:.2f}%)  orig={event_types[k]/total*100:.2f}%")
print("\nScenarios:")
for k in sorted(scenarios.keys()):
    v = scenarios2.get(k, 0)
    print(f"  {k}: {v:>6,} ({v/total2*100:.2f}%)  orig={scenarios[k]/total*100:.2f}%")
print("\nActions:")
for k in sorted(actions.keys()):
    v = actions2.get(k, 0)
    print(f"  {k}: {v:>6,} ({v/total2*100:.2f}%)  orig={actions[k]/total*100:.2f}%")
