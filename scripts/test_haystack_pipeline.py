#!/usr/bin/env python3
"""测试 Haystack Pipeline 检索功能

支持模式：
- fts: 仅全文检索（快速）
- hybrid: FTS + 向量混合检索（推荐）
"""
import argparse
import os
import sys
from pathlib import Path

# 离线模式（避免 HuggingFace Hub 网络请求）
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from haystack import Pipeline
from lancedb_haystack import LanceDBDocumentStore, LanceDBFTSRetriever, LanceDBEmbeddingRetriever
from haystack.components.joiners import DocumentJoiner


def main(query: str = "MoE架构", mode: str = "fts", top_k: int = 10) -> None:
    """执行 Haystack Pipeline 检索测试"""
    print(f"🚀 测试 Haystack Pipeline ({mode} 模式)...")

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    db_path = config['paths']['lance_db']

    # 初始化 DocumentStore（1024维，与智谱 embedding-2 兼容）
    store = LanceDBDocumentStore(
        database=db_path,
        table_name='documents',
        embedding_dims=1024
    )
    print(f"✅ LanceDB 连接成功 ({db_path})，文档数: {store.count_documents()}")

    # 构建 Pipeline
    pipeline = Pipeline()

    if mode == "hybrid":
        # 混合检索：FTS + 向量
        print("  构建 Hybrid Pipeline (FTS + 向量)...")

        # 导入智谱 Embedder
        from scripts.haystack_zhipu_embedder import ZhipuTextEmbedder

        pipeline.add_component("embedder", ZhipuTextEmbedder())
        pipeline.add_component("vector_retriever", LanceDBEmbeddingRetriever(document_store=store, top_k=top_k))
        pipeline.add_component("fts_retriever", LanceDBFTSRetriever(document_store=store, top_k=top_k))
        pipeline.add_component("joiner", DocumentJoiner())

        pipeline.connect("embedder.embedding", "vector_retriever.query_embedding")
        pipeline.connect("vector_retriever", "joiner")
        pipeline.connect("fts_retriever", "joiner")

        print("✅ Pipeline 构建成功（Hybrid 模式）")

        # 执行检索
        print(f"\n🔍 搜索: {query}")
        result = pipeline.run({
            "embedder": {"text": query},
            "fts_retriever": {"query": query}
        })
    else:
        # 仅 FTS
        print("  构建 FTS Pipeline...")
        pipeline.add_component("fts_retriever", LanceDBFTSRetriever(document_store=store, top_k=top_k))
        pipeline.add_component("joiner", DocumentJoiner())
        pipeline.connect("fts_retriever", "joiner")

        print("✅ Pipeline 构建成功（FTS 模式）")

        print(f"\n🔍 搜索: {query}")
        result = pipeline.run({"fts_retriever": {"query": query}})

    docs = result["joiner"]["documents"]
    print(f"✅ 检索结果: {len(docs)} 个文档")

    # 显示结果
    for i, doc in enumerate(docs[:5]):
        source = doc.meta.get("source", "unknown")
        content_preview = doc.content[:200] if doc.content else ""
        print(f"\n[{i+1}] 来源: {Path(source).name}")
        print(f"    内容: {content_preview}...")

    print("\n✅ Haystack Pipeline 测试完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 Haystack Pipeline")
    parser.add_argument("--query", type=str, default="MoE架构", help="搜索查询")
    parser.add_argument("--mode", type=str, default="fts", choices=["fts", "hybrid"], help="检索模式")
    parser.add_argument("--top-k", type=int, default=10, help="返回结果数量")
    args = parser.parse_args()
    main(args.query, args.mode, args.top_k)