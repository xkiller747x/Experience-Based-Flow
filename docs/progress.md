# 进度跟踪

## 当前状态：Phase 2 完成，论文数据整理中

---

## Phase 1: 基础搭建 ✅

### 1.1 仿真环境搭建 ✅
- [x] 项目目录结构创建 (2026-04-20)
- [x] 仿真数据模型定义（车辆、订单、路网）— src/simulation/models.py
- [x] 异常事件生成器 — src/simulation/generator.py
- [x] 仿真数据可视化/调试工具 — 4场景验证通过（small/medium/large/stress）

### 1.2 OR-Tools 基线求解器 ✅
- [x] VRP基础求解 — ORToolsSolver
- [x] 带时间窗的VRP (VRPTW) — AddDimension + time window SetRange
- [x] 带容量约束的VRP — AddDimensionWithVehicleCapacity (weight + volume)
- [x] 异常重规划模块 — reroute() 接口设计
- [x] 求解结果评估指标 — total_distance, total_travel_time, unassigned_order_ids

### 1.3 LLM Agent ✅
- [x] API集成（豆包/火山引擎）— src/agent/llm.py，endpoint: ark.cn-beijing.volces.com/api/coding/v3
- [x] 异常检测Prompt设计 — src/agent/prompts.py ANOMALY_DETECTION_TEMPLATE
- [x] 决策推理Prompt设计 — src/agent/prompts.py DECISION_MAKING_TEMPLATE
- [x] Few-shot示例注入 — prompts.py 内置 examples
- [x] 输出JSON结构化 — AnomalyResult / DecisionResult dataclass

### 1.4 本地量化模型部署
- [ ] llama.cpp 安装/编译 — 尚未开始
- [ ] Qwen2-7B 4-bit模型下载 — 尚未开始
- [ ] 本地推理接口封装 — 尚未开始
- [ ] 延迟/吞吐量测试 — 尚未开始

---

## Phase 2: RAG增强

### 2.1 历史case生成 ✅
- [x] case_generator.py 实现（55 cases，small×20/medium×20/large×10/stress×5）
- [x] 输出路径：data/cases/cases.jsonl
- [x] OR-Tools 批量求解验证通过
- [x] **扩展至 20 种事件类型，1000 个 case ✅**
  - 新增 16 种事件：vehicle_maintenance, fuel_shortage, driver_unavailable, traffic_congestion, road_narrowing, bridge_weight_limit, order_modify, order_update, priority_order_urgent, delivery_failure, demand_surge, demand_drop, weather_delay, natural_disaster, public_event, warehouse_delay, inventory_stockout
  - Case 分布：small=250 / medium=250 / large=250 / stress=250 = 1000
  - Outcome：success=777 (77.7%), failure=223 (22.3%)

### 2.2 case 向量化和检索 ✅
- [x] case_retriever.py — CaseRetriever 类（结构化检索 + 文本检索）
- [x] src/rag/__init__.py 更新，导出 CaseRetriever、RetrievedCase
- [x] src/agent/decision.py — recommend() 增加 retrieved_cases 参数
- [x] src/agent/prompts.py — build_decision_prompt() 追加 similar cases
- [x] src/agent/__main__.py — 插入 [6a] case 检索步骤
- [x] 端到端验证 ✅（豆包 API key 已配置，完整流程跑通）
- [x] llm.py 默认 timeout 60s → 120s（决策步骤较长，避免超时）

### 2.3 检索增强（两阶段 + Rerank） ✅
- [x] EVENT_TYPE_HIERARCHY 语义fallback映射（解决未见类型覆盖问题）
- [x] TF-IDF Index 建立（对 reasoning 字段建索引，支持语义相似度计算）
- [x] 两阶段检索：粗排 top15 → 综合打分 rerank 精选 top_k
- [x] Rerank 公式：3×事件类型 + 2×action结果 + 1.5×severity接近度 + 1×scenario匹配 + 1×tfidf语义相似度
- [x] 验收测试全部通过（traffic_congestion 不再空结果，检索稳定）

### 2.4 规则知识库 ✅
- [x] data/rules/rules.jsonl — 扩展至 20 条规则，覆盖全部 20 种事件类型
- [x] rule_retriever.py — RuleRetriever + RetrievedRule 实现
- [x] src/rag/__init__.py — 导出 Rule, RuleRetriever, RetrievedRule
- [x] decision.py — recommend() 新增 retrieved_rules 参数
- [x] prompts.py — build_decision_prompt() 追加规则注入
- [x] __main__.py — [6b] 规则检索步骤
- [x] 验收测试全部通过

---

## Phase 3: 生产部署（尚未开始）

### 3.1 API服务化
- [ ] FastAPI框架搭建
- [ ] 流式响应支持
- [ ] 异常处理/日志

### 3.2 前端界面
- [ ] 调度看板
- [ ] 异常告警列表
- [ ] 决策建议展示

### 3.3 实时数据对接
- [ ] 订单系统接口
- [ ] 车辆定位接口
- [ ] 路况数据接口

---

## Phase 3: 生产部署（进行中）

### 3.1 API服务化 ✅
- [x] FastAPI 框架搭建 — src/api/main.py，7个接口
- [x] Pydantic 请求/响应模型 — src/api/models.py
- [x] API 测试通过（/health, /simulate, /rag/cases, /rag/rules）
- [x] 豆包 API 接入，timeout 120s

### 3.2 前端界面 ✅
- [x] Streamlit 单页应用 — src/api/streamlit_app.py
- [x] 仿真场景展示（车辆/订单列表 + 指标）
- [x] 异常注入 + 检测 + 决策全流程
- [x] RAG 独立查询（Case/Rule 分 Tab）
- [x] 启动：streamlit run src/api/streamlit_app.py --server.port 8501

### 3.3 实时数据对接
- [ ] （论文场景跳过，直接使用仿真器）

---

---

## 经验模型实验结果

### V1 vs V2 vs Baseline 对比（1000 case）

| 方案 | Action 准确率 | NDCG@5 | 决策时间 |
|------|-------------|---------|---------|
| Baseline（五维打分） | 86.7% | 94.0% | 0.6ms |
| V1（action预测+奖励） | 86.7% | 90.7% | 2.5ms |
| V2（query-case相关性） | **98.1%** | **99.1%** | 9.0ms |

**结论：** V2 经验模型（相关性权重学习）与五维打分形成互补，显著提升检索质量。

---

## 技术债务 / 已知问题

| 问题 | 原因 | 状态 |
|------|------|------|
| HFSS训练OOM | 4GB GPU + 47M参数模型 | 阻塞HFSS项目 |
| RTX 3050 驱动崩溃 | 多次OOM导致 | 已重启恢复 |
| Codex Windows sandbox 故障 | 防火墙规则报错 | 已绕过：用 subagent 调用 |
| pypi 网络不通 | 学校网络限制 | 已用 logistic 环境 + 镜像安装 ortools |
| subagent 长任务超时 | 内置超时限制 | 解决：超时设到 900 秒 |

---

_最后更新：2026-04-21 00:46_