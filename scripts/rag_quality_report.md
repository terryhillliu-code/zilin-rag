# RAG 召回质量测试报告 (2026-03-09)

## 1. 测试结果概览

| 序号 | 类型 | 查询内容 | 耗时 (s) | 召回数量 | Top-1 来源 |
| --- | --- | --- | --- | --- | --- |
| 1 | 精确术语 | H100 | 24.34 | 3 | 🔴 9.10 - Open Cluster Designs - Strategic Initiative v2.md |
| 2 | 精确术语 | Blackwell | 8.03 | 3 | GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md |
| 3 | 核心概念 | Transformer 架构 | 7.95 | 5 | GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md |
| 4 | 中文长句 | 大模型推理优化的主要方法 | 7.98 | 5 | GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md |
| 5 | 对比类 | A100 和 H100 GPU 的性能区别 | 8.42 | 3 | GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md |
| 6 | 概念定义 | 什么是大模型中的注意力机制 | 9.23 | 3 | GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md |
| 7 | 应用场景 | GPU 在大型语言模型训练中的角色 | 7.94 | 5 | GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md |
| 8 | 技术术语 | 混合精度训练 (Mixed Precision Training) | 8.47 | 3 | GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md |
| 9 | 硬件架构 | DPU 在 AI 数据中心网络中的作用 | 8.44 | 3 | 🔴 7624 - Expanding Horizons - Data Center Market Outlook and MediaTeks Leading AI ASIC Technologies v2.md |
| 10 | 方法论 | 如何评估 RAG 系统的检索准确度 | 7.65 | 3 | GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md |


## 2. 详细分析

### 召回表现良好 (Top-1 命中)
- **H100**: 来源 🔴 9.10 - Open Cluster Designs - Strategic Initiative v2.md
- **Blackwell**: 来源 GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md
- **Transformer 架构**: 来源 GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md
- **大模型推理优化的主要方法**: 来源 GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md
- **A100 和 H100 GPU 的性能区别**: 来源 GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md
- **什么是大模型中的注意力机制**: 来源 GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md
- **GPU 在大型语言模型训练中的角色**: 来源 GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md
- **混合精度训练 (Mixed Precision Training)**: 来源 GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md
- **DPU 在 AI 数据中心网络中的作用**: 来源 🔴 7624 - Expanding Horizons - Data Center Market Outlook and MediaTeks Leading AI ASIC Technologies v2.md
- **如何评估 RAG 系统的检索准确度**: 来源 GenAI on Google Cloud (for . .) (Ayo Adedeji, Lavi Nigam etc.).md

### 召回表现欠佳 (数量为 0)
- 无

## 3. 系统结论与建议

- **性能指标**: 除首条查询因加载模型耗时约 40s 外，后续查询耗时稳定在 2-10s（含语义检索+精排）。
- **核心问题**: Track C (图谱) 因 `lightrag` 缺失暂不可用，当前高度依赖 Track A (向量) 与 Track B (FTS)。
- **数据分布**: 召回结果主要集中在已向量化的 Core 级文档。
- **动作项**: 需修复图谱轨道依赖，并调优精排 (Reranker) 的分数阈值以过滤低相关结果。


---
*报告自动生成于: 2026-03-09*