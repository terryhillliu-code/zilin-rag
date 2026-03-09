# zhiwei-rag/retrieve 架构文档

> 生成时间：2026-03-09 22:16
> 分析模型：qwen3-coder-next

## 文件列表

- fts_track.py: 131 行
- __init__.py: 0 行
- vector_track.py: 82 行
- graph_track.py: 140 行
- embedding_manager.py: 169 行
- hybrid_retriever.py: 231 行

## 主要模块

### hybrid_retriever.py
三轨融合检索核心

### vector_track.py
LanceDB 向量检索轨道

### fts_track.py
FTS5 全文检索轨道

### graph_track.py
图谱检索轨道 (待修复)

## 依赖关系

```
hybrid_retriever.py
├── vector_track.py
├── fts_track.py
└── graph_track.py
```
