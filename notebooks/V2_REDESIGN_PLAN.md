# V2 改造方案：从分类器到 Learned Retriever

## 一、当前问题诊断

### 现有 V2 (`ExperienceModelV2`) 的核心缺陷

1. **任务定义错误**：做的是二分类（relevant 0/1），不是检索排序
2. **特征维度太弱**：只有 10 维，全是表层匹配信号（event_type_match, scenario_match, severity_diff...）
3. **完全忽略了上下文特征**：case 里已有的 `current_load_rate`, `urgent_orders`, `time_window_pressure` 等 7 维上下文特征一个都没用
4. **依赖 GT Action**：特征里用了 `action_match`（case.action == gt_action），这是作弊信号——检索时不应该知道 GT
5. **输出形式不对**：直接输出概率，没有 Top-K 检索接口，也没有给 LLM 的结构化输出

### 现有 `CaseRetriever`（Baseline）的特点

- 两阶段：表层粗筛（规则打分）→ TF-IDF 精排
- 只用 event_type / severity / scenario + reasoning 文本
- 也没用上下文特征
- 准确率 68.7%（本质是 oracle）

---

## 二、改造目标

**把 V2 改造成 RAG 框架中的 Learned Retriever：**

```
当前物流状态（query）
    ↓
[Learned Retriever V2]  ← 改造后的模型
    ↓
Top-K 相似 case + 相关性分数 + 上下文对比
    ↓
[LLM Generator]  ← 决策层
    ↓
最终决策（action + reasoning）
```

### 核心设计原则

1. **检索器只负责"找相似 case"，不做决策**——不知道 GT action
2. **深层相似度**：不只看 event_type/scenario 表层匹配，还要看上下文压力是否相似
3. **输出对 LLM 友好**：Top-K case + 相似度分数 + 上下文差异对比

---

## 三、详细改造方案

### 3.1 特征工程重设计（核心改动）

从 10 维 → **24 维**，分三组：

#### A. 表层匹配特征（6 维）— 保留现有，去掉作弊信号
| # | 特征 | 说明 |
|---|------|------|
| 1 | event_type_match | query 和 case 的 event_type 是否相同 |
| 2 | event_type_hierarchy | 同层级事件（如 traffic_congestion ↔ traffic_accident） |
| 3 | scenario_match | 场景是否相同 |
| 4 | severity_diff | severity 差值的倒数 1/(1+|diff|) |
| 5 | vehicles_diff | 车辆数差异归一化 |
| 6 | orders_diff | 订单数差异归一化 |

#### B. 上下文相似度特征（14 维）— 新增，核心创新点
| # | 特征 | 说明 |
|---|------|------|
| 7 | load_rate_diff | 负载率差异 \|q - c\| |
| 8 | load_rate_ratio | 负载率比值 min/max |
| 9 | urgent_orders_diff | 紧急订单数差异 |
| 10 | urgent_orders_ratio | 紧急订单数比值 |
| 11 | backup_vehicles_diff | 备用车差异 |
| 12 | avg_delay_diff | 平均延迟差异 |
| 13 | avg_delay_ratio | 平均延迟比值 |
| 14 | affected_routes_diff | 受影响路线差异 |
| 15 | time_window_pressure_diff | 时间窗压力差异 |
| 16 | time_window_pressure_ratio | 时间窗压力比值 |
| 17 | customer_priority_diff | 客户优先级混合度差异 |
| 18 | customer_priority_ratio | 客户优先级混合度比值 |
| 19 | load_x_urgent | 负载率 × 紧急度 交互特征 |
| 20 | pressure_x_priority | 时间窗压力 × 优先级 交互特征 |

#### C. 全局压力特征（4 维）— query 侧的绝对值
| # | 特征 | 说明 |
|---|------|------|
| 21 | query_load_rate | query 当前的负载率 |
| 22 | query_urgent_ratio | query 的紧急订单占比 |
| 23 | query_time_pressure | query 的时间窗压力 |
| 24 | query_capacity_reserve | query 的备用车占比 |

**去掉的旧特征**：action_match（作弊）、outcome_bonus（泄露标签）、sev_x_action、ev_x_outcome、ev_code_diff、sc_code_diff

### 3.2 训练数据构造

#### 标签定义变更

**旧方案**：二分类（case.action == gt_action 且 outcome == success → 正样本）
**新方案**：**上下文相似度排序**

正样本构造策略（三选一，推荐方案 C）：

- **方案 A：Solver 一致性**
  - 对 query 调用 solver 得到 gt_action
  - case.action == gt_action 且 outcome == success → 正样本
  - 问题：仍然依赖 GT action

- **方案 B：成本相似性**
  - 对 query 和 case 各自用 solver 计算各 action 的成本向量
  - 计算两个成本向量的相似度 → 连续标签
  - 优点：完全脱离 action 标签，直接学"策略空间相似"
  - 缺点：计算量大（每个 query×case 都要跑 solver）

- **方案 C：混合标签**（推荐）
  - 正样本 = 满足以下条件之一：
    1. 表层匹配（event_type 相同 + scenario 相同）**且** 上下文相似（load_rate 差 < 0.15, severity 差 < 2）**且** outcome == success
    2. 表层不同但上下文压力模式相似（通过聚类预定义压力模式）**且** action 有效
  - 负样本 = 表层匹配但上下文差异大，或 outcome == failure
  - 优点：体现"透过表层看深层"的设计意图

#### 采样策略

- 每个 query 从全部 case 中采样：
  - 最多 20 个正样本
  - 最多 20 个硬负样本（表层匹配但上下文不同）
  - 最多 10 个随机负样本
- 总训练样本预计：30k case × 50 samples/case ≈ 150 万

### 3.3 模型结构

#### 学习目标：Learning-to-Rank（LambdaRank / GBDT 排序）

**推荐：LightGBM LambdaRank**（或保持 HistGradientBoosting 做回归）

```python
# 伪代码
import lightgbm as lgb

# 训练数据格式：每个 group 是一个 query 的所有候选 case
train_data = lgb.Dataset(X, label=y, group=query_group_sizes)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_at": [5, 10],
    "learning_rate": 0.1,
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "lambda_sparse": 0.1,
}

model = lgb.train(params, train_data, num_boost_round=300)
```

如果不想引入 LightGBM 依赖，退而求其次用 sklearn：
```python
# 回归方式：标签 = 连续相似度分数
from sklearn.ensemble import HistGradientBoostingRegressor
clf = HistGradientBoostingRegressor(...)
clf.fit(X_features, y_similarity_score)
```

### 3.4 检索接口重设计

**旧接口**：`predict_relevance(q_ev, q_sev, q_sc, cases, context)` → 返回概率列表

**新接口**：

```python
def retrieve(
    self,
    query_context: dict,       # 当前物流状态（含上下文特征）
    candidate_cases: list[Case],
    top_k: int = 5,
) -> list[RetrievedCase]:
    """
    RAG 检索：返回 Top-K 最相似的历史 case。
    
    Args:
        query_context: {
            "event_type": str,
            "severity": int,
            "scenario": str,
            "vehicles_count": int,
            "orders_count": int,
            "current_load_rate": float,
            "urgent_orders": int,
            "available_backup_vehicles": int,
            "avg_delay_minutes": float,
            "affected_routes": int,
            "time_window_pressure": float,
            "customer_priority_mix": float,
        }
        candidate_cases: 候选 case 列表
        top_k: 返回数量
    
    Returns:
        list[RetrievedCase] where RetrievedCase contains:
            - case: Case 对象
            - score: 相似度分数
            - match_reason: 匹配原因描述
            - context_diff: 上下文差异摘要（给 LLM 看）
    """
```

#### 输出增强：上下文差异摘要

```python
# 每个检索结果附带上下文对比
context_diff = {
    "load_rate": {"query": 0.82, "case": 0.78, "diff": 0.04},
    "urgent_orders": {"query": 12, "case": 10, "diff": 2},
    "time_window_pressure": {"query": 0.71, "case": 0.68, "diff": 0.03},
    "overall_similarity": 0.89,
    "key_insight": "负载率和时间窗压力接近，历史经验高度可参考"
}
```

这个 `context_diff` 会拼进 LLM prompt，帮助 LLM 理解为什么这些 case 被选中。

### 3.5 与 LLM 的协作（Prompt 设计）

检索结果拼入 prompt 的格式：

```
你是一个物流调度决策助手。当前物流状态：
- 事件: vehicle_breakdown, 严重度: 7, 场景: large
- 负载率: 0.82, 紧急订单: 12, 备用车: 2
- 平均延迟: 45min, 时间窗压力: 0.71

以下是检索到的 5 条相似历史经验：

[Case 1] 相似度: 0.92
- 上下文: 负载率 0.78(差0.04), 紧急订单 10(差2), 时间窗压力 0.68(差0.03)
- 决策: reassign_order → 成功
- 效果: 里程比 1.05, 未分配订单 -3

[Case 2] 相似度: 0.85
...

基于以上历史经验，请给出你的决策建议（action + reasoning）。
```

---

## 四、实验设计

### 对照组

| 方法 | 说明 | 特征 |
|------|------|------|
| Baseline (CaseRetriever) | 现有规则+TF-IDF | 表层 + 文本 |
| V2-Legacy | 旧 GBDT 二分类 | 10 维表层 |
| **V2-Learned (新)** | Learned Retriever + LLM | **24 维（含上下文）** |
| BM25 RAG | BM25 + LLM | 文本 |
| No-RAG LLM | 纯 LLM | 无检索 |

### 评估指标

- **检索质量**：Precision@K, Recall@K, NDCG@5, MRR
- **端到端决策**：Action Accuracy, Cost Reduction (%), NDCG@5
- **效率**：检索延迟（ms/query）

### 预期改进

| 指标 | V2-Legacy | V2-Learned（预期） |
|------|-----------|-------------------|
| Accuracy | 61.8% | ≥65% |
| NDCG@5 | 63.3% | ≥65% |
| 检索相关性 | - | 明显提升 |
| 论文叙事 | Learning-to-Rank | **RAG Learned Retriever** |

---

## 五、实现计划

### Step 1：特征工程改造
- 重写 `_query_case_features()`，24 维新特征
- 去掉 action_match / outcome_bonus 等作弊特征

### Step 2：训练数据构造
- 实现方案 C 的标签逻辑
- 采样策略：正样本 + 硬负样本 + 随机负样本

### Step 3：模型训练
- 优先 LightGBM LambdaRank（如果可以装依赖）
- 备选 HistGradientBoosting 回归

### Step 4：检索接口
- 新增 `retrieve()` 方法
- 实现 `context_diff` 上下文差异计算
- 输出 `RetrievedCase` 结构体

### Step 5：实验验证
- 先用 30k pool / 3000 queries 验证
- 对比 V2-Legacy 和 Baseline
- 确认提升后跑全量

### Step 6：LLM 集成（如需）
- 设计 prompt 模板
- 跑端到端实验

---

## 六、待确认事项

1. **标签方案**：A（Solver 一致性） / B（成本相似性） / C（混合标签）？
   - 我推荐 C，但 A 最容易实现
2. **是否引入 LightGBM**？
   - LightGBM 的 LambdaRank 是 LTR 标配，论文说服力更强
   - 如果不想加依赖，sklearn GBDT 回归也能用
3. **训练数据量级**：用多少 case 训练？全部 100 万？还是采样 30k？
   - 建议先用 30k 训练验证，效果好再扩
4. **LLM 集成**：现在就做还是先验证检索质量？
   - 建议先验证检索质量，再接 LLM

---

*Created: 2026-05-04*
*Status: 等待 Ken 确认*
