# Logistic-AI: SE-RAG for Logistics Anomaly Dispatch

物流应急调度的结构化经验增强生成（SE-RAG）研究框架。

## 核心问题

**物流异常调度是多解问题，不是分类问题。**

同一异常场景（如车辆故障、交通拥堵、订单取消）下，多个调度方案都可能合理——delay_tolerant、reroute、reassign_order
都是可接受的选择，取决调度员偏好、车队状态、隐性约束。数据中约 **48% 的场景存在多个合法调度方案**，
说明这不是数据质量问题，而是领域固有特征。

### 这对方法论意味着什么

| 方法 | 为什么不适合 | 
|---|---|
| 分类器 | 一对一的 hard label 映射，天生不适用于多解空间 |
| 纯 LLM | 缺少领域经验，69% 准确率远不够 |
| 标准 RAG (BM25/TFIDF/Dense) | 检索零散个例，缺乏结构化引导，提升有限 |
| **SE-RAG** | 规则 = 历史经验的**模式提炼**，LLM 在规则+因果链下做判断，保留多解空间 |

## SE-RAG 架构

```
                    ┌──────────────────────────────┐
                    │      离线知识构建阶段           │
                    │     (Offline Knowledge Build)  │
                    │                                │
   30k cases ──────→│  Rule Mining (44k rules)       │
                    │  ↓                             │
                    │  Causal Graph Extraction       │
                    │  ↓                             │
                    │  Weighted Pruning (~17k rules) │
                    └──────────┬───────────────────┘
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │      在线决策推理阶段           │
                    │  (Online Decision Inference)  │
                    │                                │
   Anomaly Event ──→│  Feature Extraction            │
                    │  ↓                             │
                    │  Rule Matching + Causal Signal │
                    │  ↓                             │
                    │  LLM (deepseek-v4-flash)       │
                    │  ↓                             │
                    │  JSON Decision Output          │
                    └──────────────────────────────┘
```

### 离线阶段：知识构建
1. **规则挖掘** — 从 30k 历史案例中挖掘事件特征→行动决策的关联规则，统一特征表示空间，产出 ~44k 条原始规则
2. **因果图构建** — 提取特征与行动选择的因果强度关系，帮助 LLM 理解"为什么这个特征会影响行动"
3. **规则加权** — 针对 minority action（reroute, reassign_order, adjust_capacity）加权以缓解类偏斜
4. **规则裁剪** — conf≥0.7, sup≥5 → 保留 ~17k 条高质量规则

### 在线阶段：检索增强决策
1. 输入异常事件特征（事件类型、严重等级、负载率、紧急订单数等 24 维特征）
2. 规则匹配 + 因果推理信号检索
3. 组织成 Prompt 上下文（规则证据 + 因果链 + 特征相似案例）
4. LLM 参考结构化证据输出最终调度决策

## 实验结果

### 主表（1000 queries, seed=20260428, deepseek-v4-flash）

| 排名 | 方法 | Multi-GT | Single | 平均延迟 | 错误率 |
|---|---|---|---|---|---|
| 🥇 | **SE-RAG** | **88.1%** | **61.8%** | **5.80s** | **2.8%** |
| 🥈 | bm25_rag | 73.4% | 46.4% | 9.30s | 15.5% |
| 🥉 | no_rag（纯LLM） | 69.0% | 22.3% | 7.12s | 0.4% |
| 4 | dense_rag | 68.1% | 27.0% | 9.21s | 11.1% |
| 5 | tfidf_rag | 66.0% | 37.3% | 9.40s | 17.6% |

> 全部 1000 queries（seed=20260428, deepseek-v4-flash, max_tokens=1024）。
> SE-RAG 全面领先：精度最高（+14.7pt vs bm25_rag）、速度最快（与 no_rag 相当）、错误率低于标准 RAG baseline。

### Per-Action 表现（SE-RAG vs No-RAG, 1000 queries, Multi-GT）

| Action | 占比 | No-RAG | SE-RAG | Delta |
|---|---|---|---|---|
| delay_tolerant | 48% | 61.3% | **95.7%** | **+34.4pt** 🚀 |
| ignore | 3% | 57.1% | **75.0%** | **+17.9pt** 🚀 |
| adjust_capacity | 15% | 69.1% | **75.7%** | **+6.6pt** |
| reassign_order | 23% | 79.7% | **81.5%** | **+1.8pt** |
| reroute | 11% | 88.2% | **89.1%** | **+0.9pt** |

### 关键洞察

1. **SE-RAG 的规则匹配 + 因果推理几乎不增加延迟**（5.80s vs 5.68s，差异约 2% —— 同一评估运行内比较）
2. **max_tokens 不足可导致 22pt 精度低估** — 旧版 SE-RAG（64.4%）在 max_tokens=512 下被严重低估
3. **48% GT 不一致是领域特征** — 意味着多解评估（Multi-GT）比单一匹配更能反映真实效果
4. **reroute 无退化** — Multi-GT 下单类提升逻辑更清晰
5. **标准 RAG（BM25）错误率高达 15.5%** — 为 LLM 决策增加不确定性；SE-RAG 的错误率仅 2.8%

## 目录结构

```text
logistic-ai/
├── run.py                            # 统一入口（4 个子命令）
├── config/
│   ├── llm.local.example.json        # LLM 配置模板
│   └── llm.local.json                # 本地密钥（不提交）
├── data/
│   ├── cases/                        # 历史案例（gitignored）
│   └── rules/                        # 规则知识库
│       ├── rules.jsonl               # 44k 条原始规则
│       ├── pruned/                   # 裁剪后规则集
│       └── causal_graph.json         # 因果图
├── docs/                             # 架构与设计文档
├── scripts/                          # 4 个核心脚本
│   ├── parallel_case_generator.py    # 30k/1M 案例生成
│   ├── generate_validation_queries.py# 验证集抽样
│   ├── eval_se_rag.py                # SE-RAG 对比评估
│   └── eval_standard_baselines.py    # 标准 Baseline 评估
├── src/
│   ├── agent/                        # LLM 网关与决策模块
│   ├── optimizer/                    # OR-Tools VRP 求解器
│   ├── rag/                          # 检索、规则、经验模型
│   └── simulation/                   # 仿真环境
├── record/                           # 实验日志与 paper notes
├── output/                           # 实验输出（gitignored）
└── README.md                         # 本文件
```

## 环境准备

Python 3.10+，使用 conda 管理环境：

```powershell
conda create -n logistic python=3.10
conda activate logistic
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ortools fastapi uvicorn streamlit numpy scikit-learn pydantic
```

配置环境变量（HF 镜像）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## LLM 配置

编辑 `config/llm.local.json`：

```json
{
  "provider": "openai",
  "api_key": "your-api-key-here",
  "model": "deepseek-v4-flash",
  "base_url": "https://qianweikeji.fun/v1",
  "timeout": 120
}
```

## 运行入口

所有操作通过 `run.py` 统一管理：

```powershell
# 生成 30k 案例库
python run.py generate-cases --count 30000

# 生成 1000 条验证查询（统计分布）
python run.py generate-validation --count 1000

# 运行 SE-RAG vs No-RAG 对比评估（默认 200 queries）
python run.py evaluate-se-rag --query-count 200 --seed 20260428

# 全量验证（1000 queries）
python run.py evaluate-se-rag --query-count 1000 --seed 20260428

# 跑全部标准 Baseline
python run.py evaluate-baselines --query-count 1000 --seed 20260428 --methods all

# 单方法运行（方便并行）
python run.py evaluate-baselines --query-count 1000 --method no_rag
python run.py evaluate-baselines --query-count 1000 --method bm25_rag

# 查看完整参数
python run.py evaluate-se-rag --help
```

> 内部调用 `scripts/` 下对应脚本，`run.py` 只是统一入口，不增加 runtime 开销。

## Git 提交边界

**应提交：**
- `src/` 下的业务逻辑代码
- `scripts/*.py` 实验脚本
- `data/rules/rules.jsonl` 规则知识库
- `config/*.example.json`
- `docs/`
- `README.md`

**不应提交：**
- `record/`
- `output/`
- `data/cases/`
- `*.pkl`
- `*.pyc`, `__pycache__/`
- `config/llm.local.json`

## 设计原则

1. **求解器负责约束** — OR-Tools 处理车辆路径、容量和时间窗约束
2. **RAG 提供经验证据** — 规则 + 因果图作为决策上下文
3. **LLM 保留多解空间** — 不固定唯一答案，在多解空间中做合理选择
4. **LLM 输出受控** — 固定动作集合 + JSON 格式，降低不可控风险
5. **产物本地化** — 实验结果和论文不进入仓库

## 更多文档

- `docs/architecture.md`：架构设计
- `docs/design-log.md`：关键决策
- `docs/progress.md`：实验记录
- `docs/llm-config.md`：LLM 配置说明
