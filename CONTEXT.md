# CONTEXT.md — 项目上下文

## 我是谁

我是 Delamain 🎩，Ken 的 AI 助手。这个项目是我的主人 Ken 想做的第二个方向。

## 项目背景

- **创建时间：** 2026-04-20
- **项目名：** Logistic-AI
- **路径：** D:\Code\logistic-ai
- **定位：** 基于大模型的物流调度与智能决策系统
- **硬件：** RTX 3050 (4GB) + 32GB RAM，本地部署

## 核心痛点

1. 多约束条件下调度复杂
2. 异常响应慢  
3. 全局优化难

## 技术架构

**混合架构：LLM Agent + OR-Tools + RAG**

- LLM Agent（API + 本地量化7B）→ 异常推理、决策建议
- OR-Tools → 全局最优调度求解
- RAG → 历史case检索、经验参考

**4GB GPU适配：** Qwen2-7B 4-bit量化 + llama.cpp

## 当前阶段

Phase 1 — 基础搭建（仿真环境 + OR-Tools基线 + LLM Agent）

## 关键决策记录

- ❌ 不用纯RAG做决策核心
- ❌ 不训练专用模型（无数据）
- ❌ 不用8B+模型（4GB不够）
- ✅ 混合架构：LLM推理 + OR-Tools求解 + RAG增强

## 项目结构

```
logistic-ai/
├── README.md              # 项目概述
├── docs/
│   ├── architecture.md    # 技术架构详细文档
│   ├── design-log.md      # 设计决策记录
│   └── progress.md        # 进度跟踪
├── src/                   # 源代码
├── data/                  # 数据目录
└── notebooks/             # 实验notebook
```

## 下一步行动

1. 搭建仿真环境（src/simulation/）
2. 实现 OR-Tools 基线求解器（src/optimizer/）
3. 集成 LLM Agent（src/agent/）

---

_这个文件用于跨 session 记忆项目上下文_
