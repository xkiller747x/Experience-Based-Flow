# Logistic-AI 物流智能调度系统

Logistic-AI 是一个面向物流配送场景的智能调度与异常决策原型系统。项目将仿真数据生成、OR-Tools 车辆路径优化、RAG 案例/规则检索和 LLM 决策串联起来，用于验证“运筹优化 + 检索增强 + 大模型推理”的混合调度方案。

## 当前能力

- **物流仿真**：生成车辆、订单、路网、时间窗与异常事件，支持 small / medium / large / stress 场景。
- **路径优化**：基于 Google OR-Tools 求解带容量、取送货、时间窗约束的 VRP 问题。
- **异常检测**：通过 LLM Agent 判断车辆故障、订单取消、交通事故、封路等事件是否构成调度异常。
- **RAG 决策增强**：检索历史 case 与规则库，并注入到决策 prompt 中提升建议稳定性。
- **经验模型实验**：包含 V1 / V2 经验模型、rerank 对比和论文图表数据生成脚本。
- **服务与界面**：提供 FastAPI 服务接口和 Streamlit 单页演示应用。
- **本地密钥配置**：模型名称、API Key、Base URL 统一读取 `config/llm.local.json`，真实配置不会上传 Git。

## 系统架构

```text
┌─────────────────────────────────────────────────────┐
│                    展示与服务层                      │
│        FastAPI 接口 / Streamlit 可视化演示           │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│                    LLM Agent 层                      │
│       异常检测 / 决策建议 / JSON 结果解析             │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│                    RAG 增强层                        │
│       历史 case 检索 / 规则检索 / 经验模型 rerank     │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│                  OR-Tools 优化层                     │
│       VRP 求解 / 路线重规划 / 约束验证               │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│                    仿真数据层                        │
│       车辆 / 订单 / 路网 / 事件 / 历史案例            │
└─────────────────────────────────────────────────────┘
```

## 项目结构

```text
logistic-ai/
├── config/
│   └── llm.local.example.json    # LLM 本地配置模板
├── data/
│   ├── cases/                    # 历史 case 数据
│   └── rules/                    # 规则知识库
├── docs/
│   ├── architecture.md           # 技术架构说明
│   ├── design-log.md             # 设计记录
│   ├── llm-config.md             # LLM 配置说明
│   └── progress.md               # 阶段进度与实验结果
├── notebooks/                    # 实验、评估与论文数据生成脚本
├── paper/                        # 论文 LaTeX 与图表
├── src/
│   ├── agent/                    # LLM 网关、异常检测、决策模块
│   ├── api/                      # FastAPI 服务与 Streamlit 页面
│   ├── optimizer/                # OR-Tools VRP 求解器
│   ├── rag/                      # Case/Rule 检索与经验模型
│   └── simulation/               # 仿真环境与数据模型
└── README.md
```

## 环境准备

项目使用 Python 3.10+。核心依赖包括：

- `ortools`：车辆路径优化求解
- `fastapi`、`uvicorn`：API 服务
- `streamlit`：演示界面
- `numpy`、`scikit-learn`：检索打分与经验模型
- `pydantic`：接口数据模型

如果当前环境还未安装依赖，可按需安装：

```powershell
pip install ortools fastapi uvicorn streamlit numpy scikit-learn pydantic
```

## LLM 配置

远程模型信息和 API Key 不再写入代码或环境变量，而是统一放到本地文件：

```powershell
Copy-Item config/llm.local.example.json config/llm.local.json
```

然后编辑 `config/llm.local.json`：

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
- `douban` / `openai` 需要填写 `api_key`。
- `local` 会调用本地 llama.cpp 风格接口：`{local_url}/v1/completions`。
- `config/llm.local.json` 已加入 `.gitignore`，不会被提交。
- 详细说明见 `docs/llm-config.md`。

## 快速运行

### 1. 端到端命令行演示

```powershell
python -m src.agent
```

流程包括：生成小规模仿真场景、求解初始路径、注入车辆故障、LLM 异常检测、case/rule 检索、LLM 决策建议。

### 2. FastAPI 服务

```powershell
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

主要接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/simulate` | 生成仿真场景并求解初始路线 |
| POST | `/detect` | 调用 LLM 进行异常检测 |
| POST | `/decide` | 检索 case/rule 并生成决策建议 |
| POST | `/reroute` | 在异常后重新规划路线 |
| POST | `/rag/cases` | 查询历史 case |
| POST | `/rag/rules` | 查询规则知识库 |

启动后可访问：`http://127.0.0.1:8000/docs`

### 3. Streamlit 演示界面

```powershell
streamlit run src/api/streamlit_app.py --server.port 8501
```

页面支持仿真参数配置、场景生成、异常注入、异常检测、决策建议和 RAG 检索展示。

### 4. 优化器单独演示

```powershell
python -m src.optimizer
```

用于快速验证 OR-Tools 求解器是否可用。

## 数据与实验

- 历史案例：`data/cases/*.jsonl`
- 规则库：`data/rules/rules.jsonl`
- RAG 检索实验：`notebooks/experiments.py`
- Rerank 对比实验：`notebooks/exp_rerank_comparison.py`
- V2 经验模型对比：`notebooks/exp_v2_comparison.py`
- 论文数据生成：`notebooks/paper_data_generator.py`
- 论文图表与 LaTeX：`paper/`

当前经验模型实验结果记录在 `docs/progress.md`：V2 相关性权重学习在 Action 准确率和 NDCG@5 上优于 baseline 与 V1。

## 设计要点

1. **不训练端到端大模型**：调度问题由 OR-Tools 负责，LLM 负责异常理解与决策解释。
2. **RAG 只做增强，不替代求解器**：历史案例和规则为 LLM 决策提供上下文。
3. **真实密钥本地化**：API Key 和模型信息仅保存在 `config/llm.local.json`。
4. **仿真优先**：当前项目以论文实验与原型验证为主，实时生产数据对接暂不包含。

## 当前状态

- [x] 仿真环境与场景生成
- [x] OR-Tools VRP 基线求解器
- [x] LLM 异常检测与决策流程
- [x] 历史 case 检索与规则库检索
- [x] 两阶段检索与 rerank 实验
- [x] FastAPI 服务化
- [x] Streamlit 单页演示
- [x] 本地 LLM 配置文件与 Git 忽略规则
- [ ] 实时订单/车辆/路况系统对接

## 更多文档

- `docs/architecture.md`：系统架构与设计 rationale
- `docs/design-log.md`：设计决策记录
- `docs/progress.md`：阶段进度、实验结果和已知问题
- `docs/llm-config.md`：本地模型配置说明
