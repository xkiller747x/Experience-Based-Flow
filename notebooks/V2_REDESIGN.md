# V2 经验模型重新设计 — 待定

## 核心定位
V2 = **纯 RAG 检索器**，不做任何决策，只负责"找到好的历史 case"。
决策交给 LLM 做。

## 当前问题
- V2 现在是 GBDT 二分类（relevance 0/1），本质是 Learning-to-Rank，不是 RAG
- 10 维特征全是表层匹配信号（event_type 匹配、scenario 匹配等）
- 没有利用上下文特征（load_rate, urgent_orders, time_window_pressure 等）
- 没有体现"透过表层反映深层原因"的设计意图

## 待决定的设计问题

### 1. 检索方式
- **A) 向量检索**：把 case（含上下文特征）编码成 embedding，query 也编码，cosine 相似度
- **B) 学习型打分**：GBDT/NN 打分，但加入上下文特征，输出 Top-K
- **C) 两阶段**：表层粗筛（BM25/向量）→ 上下文特征精排

### 2. "深层原因"怎么体现
- **a) 检索层面**：能匹配到"事件类型不同但上下文压力相似"的 case
- **b) 结果层面**：检索结果附带 reasoning，让 LLM 理解为什么被选中
- **c) 两者都要**

### 3. 与 LLM 的协作方式
- V2 检索 Top-K case → 拼进 prompt → LLM 做决策
- 检索结果需要什么额外信息？（reasoning？相似度分数？上下文差异？）

## 论文叙事
> RAG 检索层（V2）透过表层匹配到深层相似的历史经验 → LLM 推理层基于检索结果做决策

---
*Created: 2026-05-01, 等待 Ken 的设计决定*
