# Logistic-AI

Logistic-AI 是一个面向物流配送异常场景的智能调度原型。项目核心目标是验证一条混合路线：

```text
仿真数据 -> OR-Tools 路径优化 -> Case/Rule 检索 -> 经验模型重排 -> LLM 异常判断与决策解释
```

项目不训练端到端大模型。约束求解由 OR-Tools 负责，历史经验由 RAG 和轻量经验模型提供，LLM 只用于异常理解、决策解释和结构化 JSON 输出。

## 核心能力

- **物流仿真**：生成车辆、订单、路网、时间窗和异常事件，支持 `small`、`medium`、`large`、`stress` 四类场景。
- **路径优化**：使用 Google OR-Tools 求解带取送货、容量和时间窗约束的 VRP/VRPTW。
- **历史案例生成**：从仿真器和优化器中生成历史调度 case，作为 RAG 和经验模型训练数据。
- **Case/Rule 检索**：基于事件类型、严重程度、场景、规则触发条件和文本相似度检索历史案例和业务规则。
- **经验模型**：包含 V1 决策树预测模型和 V2 query-case 相关性模型，用于 rerank 历史案例。
- **LLM Agent**：完成异常检测、决策建议、JSON 结果解析，并可注入检索到的 case/rule。
- **API 与演示界面**：提供 FastAPI 接口和 Streamlit 单页演示。

## 目录结构

```text
logistic-ai/
├── config/
│   └── llm.local.example.json      # LLM 本地配置模板
├── data/
│   └── rules/rules.jsonl           # 规则知识库，随代码提交
├── docs/                           # 架构、配置和设计记录
├── notebooks/                      # 实验入口脚本，输出结果不提交
├── src/
│   ├── agent/                      # LLM 网关、异常检测、决策模块
│   ├── api/                        # FastAPI 和 Streamlit
│   ├── optimizer/                  # OR-Tools VRP 求解器
│   ├── rag/                        # case 生成、检索、规则检索、经验模型
│   └── simulation/                 # 仿真环境与数据模型
└── README.md
```

`paper/`、`paper_output/`、实验 JSON/CSV/图片、训练得到的 `.pkl` 模型、`data/cases/` 和 Python 缓存都属于本地产物，已在 `.gitignore` 中排除。仓库只保留项目逻辑、配置模板、规则库和可复现实验脚本。

## 环境准备

建议使用 Python 3.10+。

```powershell
pip install ortools fastapi uvicorn streamlit numpy scikit-learn pydantic
```

如果只运行仿真、case 生成、检索和经验模型实验，核心依赖是：

```powershell
pip install ortools numpy scikit-learn pydantic
```

## LLM 配置

真实模型配置只保存在本地，不提交到 Git。

```powershell
Copy-Item config/llm.local.example.json config/llm.local.json
```

编辑 `config/llm.local.json`：

```json
{
  "provider": "douban",
  "api_key": "your-api-key-here",
  "model": "doubao-seed-1-8-251228",
  "base_url": "https://ark.cn-beijing.volces.com/api/v3",
  "local_url": "http://localhost:8080",
  "timeout": 120
}
```

说明：

- `provider` 支持 `douban`、`openai`、`local`。
- 远程 provider 需要填写 `api_key`。
- `local` 用于连接本地兼容 llama.cpp/OpenAI 风格的接口。
- 详细说明见 `docs/llm-config.md`。

## 数据生成

历史案例文件默认位于：

```text
data/cases/cases.jsonl
```

该目录被 Git 忽略，需要在本地生成或从外部数据源放入。

生成默认规模 case：

```powershell
python -m src.rag.case_generator
```

默认入口会生成较大规模数据。调试时建议直接调用 `generate_all` 指定较小数量：

```powershell
python -c "from src.rag.case_generator import generate_all; generate_all(total=1000, num_workers=2, chunk_size=500)"
```

生成结果不会被 Git 追踪。

## 训练与实验逻辑

### 1. 基础检索与规则覆盖实验

```powershell
python notebooks/experiments.py
```

输出：

- `notebooks/exp1_case_coverage.json`
- `notebooks/exp2_rule_hits.json`

这些输出已被忽略，只保留在本地。

### 2. V1 rerank 对比

```powershell
python notebooks/exp_rerank_comparison.py
```

验证结构化检索与 V1 经验模型 rerank 的差异。

### 3. V2 query-case 相关性模型对比

```powershell
python notebooks/exp_v2_comparison.py
```

对比：

- `Baseline`：结构化 coarse score 检索
- `V1`：决策树预测 action 后加权重排
- `V2`：学习 query-case 相关性后重排

V2 的训练逻辑位于 `src/rag/experience_model.py` 的 `ExperienceModelV2`。

### 4. 单独训练经验模型

```powershell
python -m src.rag.experience_model
```

训练产物默认保存为 `src/rag/experience_model.pkl`，该文件已被 `.gitignore` 排除。

## 快速运行

### 命令行端到端演示

```powershell
python -m src.agent
```

流程：

1. 生成仿真场景
2. OR-Tools 求解初始路径
3. 注入异常事件
4. LLM 判断异常
5. 检索历史 case 和规则
6. LLM 输出决策建议

### OR-Tools 求解器演示

```powershell
python -m src.optimizer
```

用于确认 VRP 求解器和 OR-Tools 依赖可用。

### FastAPI 服务

```powershell
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

启动后访问：

```text
http://127.0.0.1:8000/docs
```

主要接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| POST | `/simulate` | 生成场景并求解初始路线 |
| POST | `/detect` | 调用 LLM 进行异常检测 |
| POST | `/decide` | 检索 case/rule 并生成决策建议 |
| POST | `/reroute` | 异常后重新规划路线 |
| POST | `/rag/cases` | 查询历史 case |
| POST | `/rag/rules` | 查询规则库 |

### Streamlit 演示界面

```powershell
streamlit run src/api/streamlit_app.py --server.port 8501
```

## Git 提交边界

应提交：

- `src/` 下的业务逻辑代码
- `notebooks/*.py` 实验脚本
- `data/rules/rules.jsonl`
- `config/*.example.json`
- `docs/`
- `README.md`

不应提交：

- `paper/`
- `paper_output/`
- `data/cases/`
- `notebooks/*.json`
- `notebooks/paper/`
- `*.pkl`
- `*.pyc`、`__pycache__/`
- 本地密钥配置 `config/llm.local.json`

如果某个产物已经被 Git 追踪，需要先从索引移除，但不要删除本地文件：

```powershell
git rm --cached <path>
```

## 设计原则

1. **求解器负责约束**：车辆路径、容量、取送货和时间窗由 OR-Tools 处理。
2. **RAG 提供证据**：case 和 rule 作为决策上下文，不替代优化器。
3. **经验模型做重排**：V1/V2 用于提高历史案例排序质量。
4. **LLM 输出受控**：LLM 只输出固定动作集合和 JSON 字段，降低不可控生成风险。
5. **产物本地化**：实验结果、论文、图表和训练模型不进入仓库。

## 更多文档

- `docs/architecture.md`：架构与设计 rationale
- `docs/design-log.md`：关键设计决策
- `docs/progress.md`：阶段进度和实验记录
- `docs/llm-config.md`：LLM 配置说明
