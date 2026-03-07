#!/usr/bin/env python3
"""
zhiwei-rag 命令行工具
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def cmd_search(args):
    """执行检索"""
    from retrieve.hybrid_retriever import HybridRetriever, HybridConfig
    from retrieve.embedding_manager import EmbeddingManager
    
    manager = EmbeddingManager(device='mps')
    
    config = HybridConfig(
        enable_vector=not args.no_vector,
        enable_fts=not args.no_fts,
        enable_graph=not args.no_graph,
        enable_rerank=not args.no_rerank,
        rerank_top_k=args.top_k
    )
    
    retriever = HybridRetriever(config=config, embedding_manager=manager)
    results = retriever.search(args.query, top_k=args.top_k)
    
    print(f"\n查询: {args.query}")
    print(f"精排: {'启用' if not args.no_rerank else '禁用'}")
    print(f"结果: {len(results)} 条\n")
    
    for i, r in enumerate(results):
        # 根据结果类型显示不同的分数
        if hasattr(r, 'rerank_score'):
            score_str = f"rerank={r.rerank_score:.4f}"
        else:
            score_str = f"score={r.score:.4f}"
        
        print(f"[{i+1}] [{r.track}] {score_str}")
        print(f"    来源: {r.source}")
        print(f"    内容: {r.text[:150]}...")
        print()
    
    manager.unload()


def cmd_index(args):
    """执行索引"""
    from ingest.ingest_all import main
    main()


def cmd_stats(args):
    """显示统计"""
    from ingest.lance_store import LanceStore
    
    store = LanceStore()
    print(f"LanceDB 文档数: {store.count()}")
    print(f"数据位置: {store.db_path}")


def main():
    parser = argparse.ArgumentParser(description='zhiwei-rag CLI')
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # search 命令
    search_parser = subparsers.add_parser('search', help='执行检索')
    search_parser.add_argument('query', help='查询文本')
    search_parser.add_argument('-k', '--top-k', type=int, default=5, help='返回数量')
    search_parser.add_argument('--no-vector', action='store_true', help='禁用向量检索')
    search_parser.add_argument('--no-fts', action='store_true', help='禁用 FTS 检索')
    search_parser.add_argument('--no-graph', action='store_true', help='禁用图谱检索')
    search_parser.add_argument('--no-rerank', action='store_true', help='禁用精排')
    search_parser.set_defaults(func=cmd_search)
    
    # index 命令
    index_parser = subparsers.add_parser('index', help='全量索引')
    index_parser.set_defaults(func=cmd_index)
    
    # stats 命令
    stats_parser = subparsers.add_parser('stats', help='显示统计')
    stats_parser.set_defaults(func=cmd_stats)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
