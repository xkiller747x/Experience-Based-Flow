# Baseline 实验对比框架 - 实现方案

## 目标
实现统一的评估框架，对比 6 种方法的决策准确率，用于论文消融实验。

## 6 种方法

| # | 方法 | 检索方式 | Prompt | 论文角色 |
|---|------|----------|--------|----------|
| 1 | no_rag | 无检索 | 只给异常信息 | 纯LLM下限 |
| 2 | random_rag | 随机采样5个cases | 英文通用prompt | 随机检索下限 |
| 3 | bm25_rag | BM25检索top-5 | 英文通用prompt | 传统IR baseline |
| 4 | tfidf_rag | 现有CaseRetriever top-5 | 英文通用prompt | 文本相似度baseline |
| 5 | positive_only | NegativeAware只取top-5成功cases | 英文通用prompt | 消融：去掉负样本 |
| 6 | negative_aware | 双路检索(3成功+2失败) | 中文对比式prompt | **我们的方法** |

## 需要新增/修改的文件

### 1. 新增：`src/rag/bm25_retriever.py`

```python
class BM25Retriever:
    """BM25-based case retriever for baseline comparison."""
    
    def __init__(self, cases_path: str = DEFAULT_CASES_PATH):
        # 加载 cases
        # 构建 BM25 索引，文档 = f"{event_type} {scenario} severity={severity} "
        #                          f"load_rate={current_load_rate} action={action} "
        #                          f"outcome={outcome} reasoning内容"
        # 使用 rank_bm25.BM25Okapi
        # tokenize 方式：简单的 split() 即可
    
    def retrieve(self, event_type: str, severity: int, scenario: str,
                 context: dict | None = None, top_k: int = 5) -> list[RetrievedCase]:
        # query = f"{event_type} severity={severity} {scenario}"
        # 如果有 context，追加 load_rate/pressure 等关键词
        # 返回 top-k RetrievedCase
        pass
```

依赖：`pip install rank-bm25`（在 logistic conda env 中安装）

### 2. 新增：`scripts/eval_full_comparison.py`

统一评估脚本，所有方法在同一批 query 上对比。

**流程：**
1. 加载 cases（30k）、初始化所有检索器
2. 用 SolverDrivenGroundTruth 生成 200 个 query（seed=20260428）
3. 对每个 query × 每个方法：
   - 检索（或跳过）
   - 构建 prompt
   - 调用 LLM（DeepSeek API）
   - 解析 action，对比 GT
4. 输出统计

**Query 生成方式：**
```python
# 和现有 eval_negative_aware_rag.py 一致
SCENARIOS = ["small", "medium", "large", "stress"]
ALL_EVENT_TYPES = [所有21种事件类型]
rng = random.Random(20260428)
for _ in range(200):
    scenario = rng.choice(SCENARIOS)
    event_type = rng.choice(ALL_EVENT_TYPES)
    severity = rng.randint(1, 8)
    context = solver_gt.generate_context(scenario, event_type, severity, rng)
    gt_action, _ = solver_gt.best_action(scenario, event_type, severity, context)
```

**统一 Prompt 策略：**

- **通用英文 Prompt**（用于 no_rag, random_rag, bm25_rag, tfidf_rag, positive_only）：
```
System: You are a logistics dispatch advisor. Given an anomaly and similar historical cases,
recommend the single best action.
Actions: "reroute", "ignore", "adjust_capacity", "reassign_order", "delay_tolerant"
Output JSON only: {"action": "...", "reasoning": "...", "reroute_needed": true/false}

User:
## Current Anomaly
  event_type: {event_type}
  severity: {severity}
  scenario: {scenario}
  current_load_rate: {current_load_rate}
  available_backup_vehicles: {available_backup_vehicles}
  time_window_pressure: {time_window_pressure}
  avg_delay_minutes: {avg_delay_minutes}
  urgent_orders: {urgent_orders}
  customer_priority_mix: {customer_priority_mix}

## Similar Historical Cases
{cases_block}   <-- no_rag 时这个区域写 "(none)"

Recommend the best action.
```

- **对比式中文 Prompt**（仅用于 negative_aware）：
  已有实现，和 NEGATIVE_AWARE_DECISION_TEMPLATE 一致。

**统计输出：**

```python
# 按 scenario 分组
# 按 event_type 分组
# 整体准确率
# 每个方法的 prediction distribution
# 延迟统计（检索+LLM总延迟）
```

**输出文件（output/full_comparison/）：**
- `scenario_performance.csv`: method, scenario, correct, total, accuracy_pct, avg_latency_ms
- `anomaly_type_accuracy.csv`: method, event_type, scenario, correct, total, accuracy_pct
- `overall_summary.csv`: method, correct, total, accuracy_pct, avg_latency_ms, error_count
- `raw_results.csv`: 每条query的详细结果（event_type, severity, scenario, method, gt_action, pred_action, correct, latency_ms）

**输出示例格式：**
```
=== Overall Results (200 queries) ===
Method                 Accuracy  Latency
no_rag                   XX.X%    XXXms
random_rag               XX.X%    XXXms
bm25_rag                 XX.X%    XXXms
tfidf_rag                XX.X%    XXXms
positive_only            XX.X%    XXXms
negative_aware           XX.X%    XXXms

=== By Scenario ===
Method          small    medium    large    stress
no_rag          XX.X%    XX.X%     XX.X%    XX.X%
...

=== By Event Type (top 10) ===
...
```

**运行参数：**
```python
# 命令行参数
parser.add_argument("--query-count", type=int, default=200)
parser.add_argument("--seed", type=int, default=20260428)
parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "full_comparison")
parser.add_argument("--cases-path", type=Path, default=Path(DEFAULT_CASES_PATH))
# --methods: 可选，逗号分隔，默认全部
parser.add_argument("--methods", type=str, default="all")
```

## 实现约束

1. 使用 `D:\Conda\mini\envs\logistic\python.exe`
2. 先在 logistic env 中安装 rank-bm25：`pip install rank-bm25 -i https://pypi.tuna.tsinghua.edu.cn/simple`
3. BM25Retriever 不修改任何现有代码，纯新增
4. 统一使用 `LLMGateway`（从 config/llm.local.json 读配置）
5. 所有方法的 LLM 调用使用 temperature=0.0（LLMGateway 默认值）
6. 对 LLM 返回的 action 解析统一用 `extract_action()` 函数
7. 每个 query 对所有方法顺序执行（不要并行，避免 API 限流）
8. 每 20 条 query 打印一次进度
9. cases_path 用 `D:/Code/logistic-ai/data/cases/cases_30k.jsonl`
10. **不要修改任何现有文件**，只新增 bm25_retriever.py 和 eval_full_comparison.py

## 验证标准

1. `BM25Retriever` 能独立加载和检索，返回 list[RetrievedCase]
2. `eval_full_comparison.py --query-count 10` 能在 2 分钟内跑完
3. 输出文件格式正确，能被后续脚本读取
4. 6 个方法都有结果，error_count < 5%
