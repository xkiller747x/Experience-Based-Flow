# 负样本感知 RAG 实现方案

## 目标
在现有 Learned-RAG 基础上，实现双路检索（正例+负例）和对比式 Prompt，让 LLM 利用失败案例的信号提升决策准确率。

## 需要修改/新增的文件

### 1. 新增文件：`src/rag/negative_aware_retriever.py`

实现 `NegativeAwareRetriever` 类：

```python
class NegativeAwareRetriever:
    """
    双路检索器：分别检索相似 context 的成功和失败案例。
    
    检索流程：
    1. 从所有 cases 中筛选与 query 相同 (event_type, scenario) 的候选
    2. 计算 context 相似度（基于以下特征的加权欧式距离）：
       - severity_diff (权重 3.0)
       - current_load_rate_diff (权重 2.5)
       - available_backup_vehicles_diff (权重 2.0)
       - time_window_pressure_diff (权重 2.0)
       - avg_delay_minutes_diff (权重 1.5)
       - customer_priority_mix_diff (权重 1.0)
       - urgent_orders_diff (权重 1.0)
    3. 在成功案例中取 top-3（正例）
    4. 在失败案例中取 top-2（负例）
    5. 返回结构化结果
    """
```

**核心方法签名：**
```python
@dataclass
class DualRetrievalResult:
    success_cases: list[RetrievedCase]   # top-3 成功案例
    failure_cases: list[RetrievedCase]   # top-2 失败案例
    context_similarity: float            # 整体 context 匹配度

class NegativeAwareRetriever:
    def __init__(self, cases_path: str = "D:/Code/logistic-ai/data/cases/cases_30k.jsonl"):
        # 加载 cases，分别建立 success / failure 索引
        pass
    
    def retrieve(self, event_type: str, severity: int, scenario: str,
                 context: dict, top_success: int = 3, top_failure: int = 2) -> DualRetrievalResult:
        """
        双路检索。
        context 字典包含: current_load_rate, urgent_orders, available_backup_vehicles,
                          avg_delay_minutes, affected_routes, time_window_pressure, customer_priority_mix
        """
        pass
    
    def _context_distance(self, query_ctx: dict, case_ctx: dict) -> float:
        """加权欧式距离，返回距离越小越相似"""
        pass
```

**context_distance 具体实现：**
- 所有特征先归一化到 [0, 1] 区间（severity 除以 10，urgent_orders 除以 max_orders 等）
- 按权重计算加权距离
- 距离越小 = context 越相似

### 2. 修改文件：`src/agent/prompts.py`

新增 `NEGATIVE_AWARE_DECISION_TEMPLATE`，替换当前 `DECISION_MAKING_TEMPLATE`：

**system_template 核心内容：**
```
你是一个物流调度顾问。基于历史案例的经验和教训，为当前异常推荐最佳行动。

可选行动：
- "reroute": 重新优化受影响车辆路线
- "ignore": 无需处理
- "adjust_capacity": 调整车辆容量分配
- "reassign_order": 将订单重新分配给其他车辆
- "delay_tolerant": 接受延迟

关键规则：
1. 优先采取【成功案例】中高频出现的行动
2. 严格避免采取【失败案例】中出现过的行动
3. 如果成功案例的行动建议不一致，结合当前 severity 和 load_rate 判断
4. 只输出 JSON，不要有其他文字
```

**user_template 核心结构：**
```
## 当前异常
  event_type: {event_type}
  severity: {severity}
  scenario: {scenario}
  current_load_rate: {current_load_rate}
  available_backup_vehicles: {available_backup_vehicles}
  time_window_pressure: {time_window_pressure}

## ✅ 成功经验（类似情况下成功的案例）
{success_cases_block}

## ❌ 失败教训（类似情况下失败的案例，请避免这些行动）
{failure_cases_block}

请基于以上正负经验推荐最佳行动。
输出 JSON: {{"action": "...", "reasoning": "...", "reroute_needed": true/false}}
```

**同时修改 `build_decision_prompt` 函数**，新增参数支持：
```python
def build_decision_prompt(
    anomaly_result: dict,
    route_plan,
    vehicles: list,
    orders: list,
    retrieved_cases: list[RetrievedCase] | None = None,
    retrieved_rules: list[RetrievedRule] | None = None,
    dual_retrieval_result: DualRetrievalResult | None = None,  # 新增
) -> tuple[str, str]:
```

当 `dual_retrieval_result` 不为 None 时，使用新模板；否则使用原模板。

### 3. 修改文件：`src/agent/decision.py`

在 `DecisionMaker.recommend` 方法中：
- 新增 `dual_retrieval_result` 参数
- 传递给 `build_decision_prompt`

### 4. 新增文件：`scripts/eval_negative_aware_rag.py`

评估脚本，流程：
1. 加载 30k cases（训练集也是检索库）
2. 用 solver_ground_truth 生成 200 个 query（随机采样）
3. 对每个 query：
   a. 用 NegativeAwareRetriever 做双路检索
   b. 用 LLM（DeepSeek API）做决策
   c. 对比 GT action 计算准确率
4. 同时跑 baseline（现有 CaseRetriever + LLM）做对比
5. 输出到 `output/negative_aware_rag/` 目录
6. 分别统计：整体准确率、按 event_type 准确率、按 scenario 准确率
7. 统计 failure case 检索质量：检索到的负例 context 距离分布

**输出格式**同现有的 scenario_performance.csv 和 anomaly_type_accuracy.csv。

### 5. 新增文件：`scripts/quick_eval_negative_aware.py`

快速测试脚本（不调 LLM，只测检索质量）：
- 200 query
- 对比三种检索策略检索到的正负例质量
- 输出：failure case 检索命中率、context 距离分布、正负例 action 分歧统计

## 实现约束

1. 所有代码放在 `D:\Code\logistic-ai\` 项目下
2. Python 环境使用 `D:\Conda\mini\envs\logistic\python.exe`
3. 遵循现有代码风格（dataclass、类型标注、docstring）
4. 不要修改现有 `CaseRetriever`、`LearnedRetrieverV2` 的代码，新代码是独立的
5. `prompts.py` 的修改要保持向后兼容（不影响原有模板）
6. context 归一化参数参考 solver_ground_truth.py 中的 SCENARIO_PROFILES
7. JSONL 路径统一使用 `D:/Code/logistic-ai/data/cases/cases_30k.jsonl`
8. DeepSeek API 配置从 `config/llm.local.json` 读取，使用现有的 `LLMGateway` 类

## 验证标准

1. `quick_eval_negative_aware.py` 能独立运行，输出检索质量统计
2. `eval_negative_aware_rag.py` 能端到端跑通 200 query
3. 目标：Learned-RAG 准确率从 37.5% 提升到 50%+（利用负样本信号）
