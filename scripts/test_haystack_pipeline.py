#!/usr/bin/env python3
"""测试 Haystack Pipeline 检索功能"""
import argparse
from pathlib import Path
import yaml
from haystack import Pipeline
from lancedb_haystack import LanceDBDocumentStore, LanceDBFTSRetriever
from haystack.components.joiners import DocumentJoiner


def main(query: str = "MoE架构") -> None:
    """执行 Haystack Pipeline 检索测试"""
    print(f"🚀 测试 Haystack Pipeline...")

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    db_path = config['paths']['lance_db']

    # 现有数据使用智谱 embedding-2 (1024维)，Haystack 默认 embedder 不兼容，仅使用 FTS
    store = LanceDBDocumentStore(
        database=db_path,
        table_name='documents',
        embedding_dims=1024
    )
    print(f"✅ LanceDB 连接成功 ({db_path})，文档数: {store.count_documents()}")

    pipeline = Pipeline()
    pipeline.add_component("fts_retriever", LanceDBFTSRetriever(document_store=store, top_k=10))
    pipeline.add_component("joiner", DocumentJoiner())
    pipeline.connect("fts_retriever", "joiner")

    print("✅ Pipeline 构建成功（FTS模式）")

    print(f"\n🔍 搜索: {query}")

    result = pipeline.run({"fts_retriever": {"query": query}})

    docs = result["joiner"]["documents"]
    print(f"✅ 检索结果: {len(docs)} 个文档")

    for i, doc in enumerate(docs[:5]):
        source = doc.meta.get("source", "unknown")
        content_preview = doc.content[:200] if doc.content else ""
        print(f"\n[{i+1}] 来源: {Path(source).name}")
        print(f"    内容: {content_preview}...")

    print("\n✅ Haystack Pipeline 测试完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 Haystack Pipeline")
    parser.add_argument("--query", type=str, default="MoE架构", help="搜索查询")
    args = parser.parse_args()
    main(args.query)