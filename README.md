# zhiwei-rag

知微系统的混合检索增强模块。

## 功能

- **三轨检索**：向量 (LanceDB) + 全文 (FTS5) + 图谱 (LightRAG)
- **精排层**：bge-reranker-base Cross-Encoder
- **语义切分**：Markdown 标题感知 + 元数据注入
- **本地 Embedding**：bge-large-zh-v1.5 + MPS 加速
- **按需加载**：空闲自动释放内存

## 使用

### 命令行

```bash
# 检索
python cli.py search "深度学习" -k 5

# 不带精排
python cli.py search "深度学习" -k 5 --no-rerank

# 全量索引
python cli.py index

# 统计
python cli.py stats
```

### Python API

```python
from api import get_rag

rag = get_rag()

# 检索
results = rag.retrieve("什么是 Transformer？", top_k=5)

# 获取上下文
context = rag.get_context("什么是 Transformer？")

# 构建完整 Prompt
prompt, results = rag.retrieve_and_build_context(
    "什么是 Transformer？",
    template="qa"
)

# 清理
rag.cleanup()
```

### 子进程桥接
供其他 Python 进程调用（避免 asyncio 冲突）：

```bash
# 获取上下文
python bridge.py context "查询文本" --top-k 5

# 获取 JSON 格式检索结果
python bridge.py retrieve "查询文本" --top-k 5

# 获取完整 Prompt
python bridge.py prompt "查询文本" --template qa
```

## 目录结构

```text
zhiwei-rag/
├── api.py              # 统一 API
├── bridge.py           # 子进程桥接
├── cli.py              # 命令行工具
├── config.yaml         # 配置文件
├── ingest/             # 数据摄入
│   ├── semantic_splitter.py
│   ├── lance_store.py
│   └── ingest_all.py
├── retrieve/           # 检索
│   ├── embedding_manager.py
│   ├── vector_track.py
│   ├── fts_track.py
│   ├── graph_track.py
│   └── hybrid_retriever.py
├── rank/               # 精排
│   └── reranker.py
├── generate/           # 生成
│   ├── context_builder.py
│   └── prompt_templates/
└── data/               # 数据
    ├── lance_db/
    └── models/
```

## 配置
编辑 config.yaml 调整参数。

## 依赖
- lancedb
- sentence-transformers
- transformers
- torch (MPS 支持)
