#!/usr/bin/env python3
"""批量入库脚本 - V9.0 优化版

核心优化：模型仅加载一次，流式处理多文件，支持进度条。
预计提速 10-20 倍。
"""
import os, sys, time, argparse, yaml, fcntl, json
from pathlib import Path

# 设置离线模式，使用本地缓存
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# 环境初始化
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingest.semantic_splitter import SemanticSplitter
from ingest.lance_store import LanceStore
from retrieve.embedding_manager import EmbeddingManager
from ingest.ingest_all import chunks_to_documents

def ingest_batch(file_list, prefix="vault:", vlm_enabled=False):
    lock_file = open("/tmp/zhiwei-ingest.lock", "w")
    try:
        print(f"🔐 请求系统锁并加载模型 (PID: {os.getpid()})...")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        # 加载配置
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f: config = yaml.safe_load(f)

        # 【核心优化】：模型一次性预加载
        emb = EmbeddingManager(model_name=config['embedding']['model_name'], device=config['embedding']['device'])
        emb.preload()

        store = LanceStore(db_path=config['paths']['lance_db'], embedding_manager=emb)
        splitter = SemanticSplitter(**config['splitter'])

        stats = {"success": 0, "fail": 0}
        total = len(file_list)

        print(f"📊 开始处理 {total} 个文件...\n", flush=True)

        for i, filepath in enumerate(file_list):
            p = Path(filepath).resolve()
            print(f"🚀 [{i+1}/{total}] 处理: {p.name}...", flush=True)
            try:
                t0 = time.time()
                chunks = splitter.split_file(p)
                if chunks:
                    store.delete_by_source(str(p))
                    embeddings = emb.encode([c.text for c in chunks])
                    docs = chunks_to_documents(chunks, embeddings, source_prefix=prefix)
                    store.add_documents(docs)
                stats["success"] += 1
                elapsed = time.time() - t0
                print(f"   ✅ 完成 (耗时 {elapsed:.1f}s, {len(chunks) if chunks else 0} chunks)", flush=True)
            except Exception as e:
                print(f"   ❌ 失败: {e}", flush=True)
                stats["fail"] += 1

        print(f"\n🎉 批量处理完成! 成功: {stats['success']}, 失败: {stats['fail']}", flush=True)
        return stats
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量入库脚本 V9.0")
    parser.add_argument("--files", nargs="+", help="直接传入文件路径列表")
    parser.add_argument("--list", type=str, help="传入包含文件路径的 JSON/TXT 列表文件")
    args = parser.parse_args()

    work_list = []
    if args.files: work_list = args.files
    elif args.list:
        with open(args.list, 'r') as f:
            work_list = [line.strip() for line in f if line.strip()]

    if work_list:
        ingest_batch(work_list)
    else:
        print("❌ 请提供 --files 或 --list 参数")
        sys.exit(1)