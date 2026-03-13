# zhiwei-rag/retrieve 架构文档

> 更新时间：2026-03-13
> FTS-001: FTS 轨道已从 klib.db 迁移到 LanceDB

## 文件列表

- vector_track.py: 向量检索 + FTS（LanceDB）
- graph_track.py: 图谱检索轨道
- embedding_manager.py: Embedding 管理器
- hybrid_retriever.py: 三轨融合检索核心

## 主要模块

### hybrid_retriever.py
三轨融合检索核心

### vector_track.py
LanceDB 向量检索 + 全文检索轨道
- `search()`: 向量语义检索
- `search_fts()`: jieba 分词 + FTS 关键词检索

### graph_track.py
图谱检索轨道 (LightRAG)

## 依赖关系

```
hybrid_retriever.py
├── vector_track.py  (向量 + FTS)
└── graph_track.py
```