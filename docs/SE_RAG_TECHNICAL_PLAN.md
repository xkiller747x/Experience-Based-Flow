# SE-RAG: Structured Experience-enhanced RAG 技术方案

## 1. 核心思想

传统 RAG 直接检索原始案例（表面相似），SE-RAG 先离线提炼结构化经验（决策规则 + 因果关系），再在线匹配并辅助 LLM 做可解释决策。

**论文定位：** 可解释性 + 泛化性的决策辅助 RAG 框架

## 2. 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   离线阶段（Offline）                  │
│                                                      │
│  Raw Cases (30k)                                     │
│      ↓                                               │
│  Step 1: 决策规则挖掘（Decision Rule Mining）          │
│      → IF-THEN 规则 + 置信度 + 支撑样本               │
│      ↓                                               │
│  Step 2: 因果关系提取（Causal Relation Extraction）    │
│      → 因素→因素 / 因素→决策 的影响关系                │
│      ↓                                               │
│  结构化经验库（Structured Experience Base）            │
│    - 决策规则表                                       │
│    - 因果关系图                                       │
│    - 规则→支撑案例 映射                               │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   在线阶段（Online）                   │
│                                                      │
│  Query (event_type, severity, context)               │
│      ↓                                               │
│  Step 3: 规则匹配（Rule Matching）                    │
│      → 找到触发的规则 + 未触发的对比规则               │
│      ↓                                               │
│  Step 4: 证据组装（Evidence Assembly）                │
│      → 触发规则 + 支撑案例 + 因果链 + 对比规则         │
│      ↓                                               │
│  Step 5: LLM 因果推理（Causal Reasoning）             │
│      → 带因果解释的决策 + 为什么不选其他 action        │
└─────────────────────────────────────────────────────┘
```

## 3. 各步骤详细设计

### Step 1: 决策规则挖掘

**方法：** 分组统计 + 决策树

对每个 (event_type)，按 context 特征分桶统计：

```python
# 伪代码
for event_type in all_event_types:
    cases = filter(cases, event_type=event_type)
    # 对连续特征离散化
    # severity: low(1-3), mid(4-6), high(7-8)
    # load_rate: low(≤0.4), mid(0.4-0.7), high(>0.7)
    # backup_vehicles: none(0), few(1-2), enough(≥3)
    # time_window_pressure: low(≤0.4), high(>0.4)
    
    # 统计每个分桶下的 action 分布
    for bucket in all_buckets:
        action_dist = count(cases in bucket) → {action: count}
        dominant_action = argmax(action_dist)
        confidence = dominant_count / total_in_bucket
        support_cases = sample(cases in bucket, k=3)
        
        if confidence >= 0.7 and total_in_bucket >= 10:
            rules.append(Rule(
                condition=bucket,
                action=dominant_action,
                confidence=confidence,
                support_count=total_in_bucket,
                support_cases=support_cases,
            ))
```

**输出示例：**
```
Rule #1: vehicle_breakdown + severity=high + load_rate=high
  → reassign_order (conf=0.94, support=312 cases)

Rule #2: vehicle_breakdown + severity=high + load_rate=low
  → adjust_capacity (conf=0.91, support=87 cases)

Rule #3: vehicle_breakdown + severity=low
  → reroute (conf=0.88, support=203 cases)
```

**泛化性：** 算法只依赖 (特征, action, outcome) 三元组，不依赖物流领域的具体特征含义。

### Step 2: 因果关系提取

**方法：** 统计因果发现（条件概率差异）

```python
# 对于每个因素对 (factor_A, factor_B)，计算：
# P(action=X | factor_A=v1, factor_B) vs P(action=X | factor_A=v2, factor_B)
# 如果差异显著，说明 factor_A 对 action 有因果影响（控制 factor_B）

for factor_A in context_features:
    for factor_B in context_features:
        if factor_A == factor_B: continue
        # 计算条件互信息或概率差异
        causal_strength = conditional_mutual_info(factor_A, action, given=factor_B)
        if causal_strength > threshold:
            causal_graph.add_edge(factor_A, action, weight=causal_strength)
```

**输出：** 一张因果图
```
severity ──(0.82)──→ reassign_order
load_rate ──(0.71)──→ adjust_capacity
backup_vehicles ──(-0.65)──→ reroute  # 负相关：backup多→不需要reroute
time_window_pressure ──(0.58)──→ delay_tolerant
```

**泛化性：** 因果发现算法是领域无关的。

### Step 3: 规则匹配（在线）

给定 query (event_type, severity, context)：

1. **精确匹配：** 找到所有 condition 被满足的规则
2. **部分匹配：** 找到最接近的规则（允许 1-2 个特征不同）
3. **对比规则：** 找到条件相似但 action 不同的规则（用于解释"为什么不是 Y"）

```python
def match_rules(query, rule_base):
    matched = exact_match(query, rule_base)  # 精确触发
    near = near_match(query, rule_base, max_diff=1)  # 近似触发
    contrast = find_contrast(matched, rule_base)  # 条件相似但结论不同
    return matched, near, contrast
```

### Step 4: 证据组装

将匹配结果组装成结构化证据：

```json
{
  "triggered_rules": [
    {
      "rule": "vehicle_breakdown + severity=high + load_rate=high → reassign_order",
      "confidence": 0.94,
      "support_cases": [案例摘要...]
    }
  ],
  "causal_chain": [
    "severity=7 导致大量车辆不可用",
    "load_rate=0.85 表示剩余车辆接近满载",
    "两者叠加 → 需要重新分配订单而非简单重路由"
  ],
  "contrast_rules": [
    {
      "rule": "vehicle_breakdown + severity=high + load_rate=low → adjust_capacity",
      "reason": "load_rate 低时剩余容量充足，调整容量即可"
    }
  ]
}
```

### Step 5: LLM 因果推理 Prompt

```
你是一个物流调度决策专家。基于结构化经验库的检索结果，为当前异常做决策。

## 当前异常
event_type: vehicle_breakdown, severity: 7, scenario: large
current_load_rate: 0.85, available_backup_vehicles: 1, ...

## 触发的决策规则
Rule #1: severity=high + load_rate=high → reassign_order (置信度94%, 312条案例支持)

## 因果推理链
- severity=7 → 预计 3-5 辆车受影响
- load_rate=0.85 → 剩余运力仅15%，无法通过 reroute 消化
- backup_vehicles=1 → 备用车不足，不能靠 adjust_capacity 解决

## 对比（为什么不是其他 action）
- 不选 reroute: load_rate 过高，剩余路线容量不足 (Rule #2, 置信度91%)
- 不选 delay_tolerant: time_window_pressure=0.78, 延迟代价过高

请基于以上证据做出决策。
输出 JSON: {"action": "...", "reasoning": "...", "reroute_needed": true/false}
```

## 4. 实验设计

### Baselines
1. **No-RAG** — 纯 LLM，不提供任何历史信息
2. **Naive RAG** — 传统向量检索 + LLM
3. **BM25-RAG** — 关键词检索 + LLM
4. **Positive-Only RAG** — 只检索成功案例
5. **SE-RAG (Ours)** — 结构化经验增强 RAG

### 评估指标
1. **决策准确率** — Action Accuracy
2. **可解释性评分** — 人工/LLM 评估 reasoning 的因果合理性（1-5分）
3. **泛化性测试** — 在不同 scenario 分布下测试（训练用 small+medium，测试用 large+stress）
4. **Few-shot 泛化** — 只用 1k cases 建规则库，测试准确率下降幅度

### 预期结果
- SE-RAG 在准确率上 ≥ Positive-Only RAG
- SE-RAG 在可解释性上显著优于所有 baseline
- SE-RAG 在泛化性（跨 scenario / few-shot）上优于传统 RAG

## 5. 论文叙事

> **标题方向：** SE-RAG: Structured Experience-Enhanced Retrieval-Augmented Generation for Explainable Logistics Decision-Making
>
> **核心贡献：**
> 1. 提出 SE-RAG 框架，将 RAG 从"检索原文"升级为"检索结构化经验"
> 2. 设计离线规则挖掘 + 因果发现 + 在线规则匹配 + LLM 因果推理的完整流程
> 3. 在物流决策场景验证了可解释性和泛化性优势
>
> **故事线：**
> 传统 RAG 检索表面相似的案例 → LLM 模仿案例的 action
> → 问题：案例的 outcome 不完全由 action 决定，表面相似≠决策参考价值高
> → 我们的方法：先提炼"在什么条件下该做什么"的规则和因果链
> → LLM 基于因果推理做决策，而非模仿案例
> → 结果：更准、更可解释、更泛化

## 6. 实现计划

### Phase 1: 规则挖掘（1-2天）
- [ ] 实现 DecisionRuleMiner
- [ ] 对 30k cases 提炼规则
- [ ] 输出规则表 + 统计报告

### Phase 2: 因果发现（1-2天）
- [ ] 实现 CausalRelationExtractor
- [ ] 构建因果关系图
- [ ] 可视化验证

### Phase 3: 在线检索 + LLM（2-3天）
- [ ] 实现 RuleMatcher（精确+近似+对比）
- [ ] 实现 EvidenceAssembler
- [ ] 设计因果推理 Prompt
- [ ] 集成到 eval_full_comparison.py

### Phase 4: 实验评估（2-3天）
- [ ] 跑全部 baseline 对比
- [ ] 可解释性评估
- [ ] 泛化性测试
- [ ] 生成论文图表

### 总预估：1-2 周

---
*Created: 2026-05-08*
*Status: 待确认，等 Ken 确认后开始实施*
