"""验证经验模型的效果和使用方式。"""

import sys
sys.path.insert(0, "D:/Code/logistic-ai")

from src.rag import ExperienceModel, CaseRetriever

# 加载模型
model = ExperienceModel.load("D:/Code/logistic-ai/src/rag/experience_model.pkl")

# 测试预测
print("=" * 60)
print("  Experience Model — Prediction Tests")
print("=" * 60)

tests = [
    ("vehicle_breakdown", 5, "small"),
    ("vehicle_breakdown", 1, "small"),
    ("order_cancel", 3, "medium"),
    ("order_cancel", 1, "small"),
    ("traffic_congestion", 4, "large"),
    ("demand_surge", 3, "stress"),
    ("weather_delay", 5, "small"),
]

for ev, sev, sc in tests:
    pred = model.predict(ev, sev, sc)
    print(f"\n[{ev}] severity={sev} scenario={sc}")
    print(f"  action={pred['action']} (conf={pred['action_confidence']:.2f})")
    print(f"  outcome={pred['outcome']} (conf={pred['outcome_confidence']:.2f})")
    print(f"  distance_ratio={pred['distance_ratio']:.3f}")
    print(f"  unassigned_delta={pred['unassigned_delta']}")

print("\n[OK] Experience model tests passed!")