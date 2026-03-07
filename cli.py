#!/usr/bin/env python3
"""
zhiwei-rag 命令行工具
"""
import sys
import argparse
import os
from pathlib import Path

# 确保导入 path 正确
sys.path.insert(0, str(Path(__file__).parent))


def cmd_search(args):
    """执行检索"""
    from retrieve.hybrid_retriever import HybridRetriever, HybridConfig
    from retrieve.embedding_manager import EmbeddingManager
    
    manager = EmbeddingManager(device='mps')
    
    config = HybridConfig(
        enable_vector=not args.no_vector,
        enable_fts=not args.no_fts,
        enable_graph=not args.no_graph
    )
    
    retriever = HybridRetriever(config=config, embedding_manager=manager)
    results = retriever.search(args.query, top_k=args.top_k)
    
    print(f"\n查询文本: {args.query}")
    print(f"召回结果: {len(results)} 条\n")
    
    for i, r in enumerate(results):
        print(f"[{i+1}] [{r.track}] 综合分数={r.score:.4f}")
        print(f"    来源: {r.source}")
        print(f"    摘要: {r.text[:150].replace('\n', ' ')}...")
        print("-" * 40)
    
    manager.unload()


def cmd_index(args):
    """执行索引"""
    from ingest.ingest_all import main
    main()


def cmd_stats(args):
    """显示统计"""
    from ingest.lance_store import LanceStore
    
    store = LanceStore()
    count = store.count()
    print(f"LanceDB 索引文档数: {count}")
    print(f"数据库路径: {store.db_path}")


def main():
    parser = argparse.ArgumentParser(description='zhiwei-rag 面向未来的智能检索工具')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # search 命令
    search_parser = subparsers.add_parser('search', help='执行多轨混合检索')
    search_parser.add_argument('query', help='查询文本')
    search_parser.add_argument('-k', '--top-k', type=int, default=5, help='返回结果数量')
    search_parser.add_argument('--no-vector', action='store_true', help='禁用向量索引')
    search_parser.add_argument('--no-fts', action='store_true', help='禁用全文索引')
    search_parser.add_argument('--no-graph', action='store_true', help='禁用图谱索引')
    search_parser.set_defaults(func=cmd_search)
    
    # index 命令
    index_parser = subparsers.add_parser('index', help='执行全量入库索引')
    index_parser.set_defaults(func=cmd_index)
    
    # stats 命令
    stats_parser = subparsers.add_parser('stats', help='系统状态和统计')
    stats_parser.set_defaults(func=cmd_stats)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
