#!/usr/bin/env python3
"""
增量入库脚本：
针对单篇文档（如刚下载的 ArXiv 论文或刚产出的研报），执行实时入库，消除搜索断层。
"""
import os
import sys
import time
import argparse
from pathlib import Path
import yaml
import json

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.semantic_splitter import SemanticSplitter
from ingest.lance_store import LanceStore
from retrieve.embedding_manager import EmbeddingManager
from ingest.ingest_all import chunks_to_documents
from ingest.pdf_multimodal_ingest import ingest_pdf_multimodal
from ingest.vlm_describer import VLMDescriber

def send_feishu_alert(filename, error_msg):
    """通过调度器队列发送飞书告警 (V4.2)"""
    try:
        from datetime import datetime, timezone
        pending_dir = Path.home() / "zhiwei-scheduler" / "outputs" / "artifacts" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        
        job_id = f"ingest_fail_{int(time.time())}"
        payload = {
            "job_id": job_id,
            "task": "ingest_alert",
            "content": f"# 🚨 论文入库失败\n\n**文件**: {filename}\n**错误**: {error_msg}\n\n> 请检查 `zhiwei-rag` 日志或手动尝试 `/research`。",
            "push_targets": ["feishu"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "retries": 0,
            "last_error": None,
            "metadata": {"type": "error", "source": "ingest_incremental"}
        }
        
        target_path = pending_dir / f"{job_id}.json"
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"📡 故障告警已落盘: {target_path.name}")
    except Exception as e:
        print(f"❌ 发送告警失败: {e}")

async def ingest_core(filepath, prefix="vault:", vlm_enabled=True):
    import fcntl
    lock_file_path = Path("/tmp/zhiwei-ingest.lock")
    lock_file = None

    try:
        # 确保锁文件存在
        lock_file_path.touch(exist_ok=True)
        lock_file = open(lock_file_path, "w")
        
        print(f"⏳ 正在请求入库锁 (PID: {os.getpid()})...")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        print(f"🔐 已锁定资源，处理: {filepath.name}")

        # 加载配置
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        print(f"🔄 开始增量索引: {filepath.name}...")

        # 初始化组件
        embedding_manager = EmbeddingManager(
            model_name=config['embedding']['model_name'],
            device=config['embedding']['device'],
            idle_timeout=config['embedding']['idle_timeout']
        )
        
        store = LanceStore(
            db_path=config['paths']['lance_db'],
            embedding_manager=embedding_manager
        )
        
        splitter = SemanticSplitter(
            max_chunk_tokens=config['splitter']['max_chunk_tokens'],
            min_chunk_chars=config['splitter']['min_chunk_chars'],
            fallback_to_paragraph=config['splitter']['fallback_to_paragraph']
        )

        t0 = time.time()
        
        # 预加载与建表检查
        embedding_manager.preload()
        store.create_table(dimension=embedding_manager.dimension)

        # 切分文本文档
        chunks = splitter.split_file(filepath)
        if chunks:
            # 清理同源旧数据
            store.delete_by_source(str(filepath))
            
            # 执行编码
            texts = [c.text for c in chunks]
            print(f"🧩 提取到 {len(texts)} 个文本块，开始向量化...")
            embeddings = embedding_manager.encode(texts)

            # 存入数据库
            docs = chunks_to_documents(chunks, embeddings, source_prefix=prefix)
            store.add_documents(docs)
            print(f"   ✅ 文本入库完成: {len(docs)} 个 Chunk")

        # === 新增：多模态 PDF 深度分析 (Research V5.0) ===
        pdf_path = filepath.with_suffix(".pdf")
        if not pdf_path.exists():
            # 搜索 Library
            library_root = Path.home() / "Documents" / "Library"
            potential_pdfs = list(library_root.rglob(f"{filepath.stem}.pdf"))
            if potential_pdfs: pdf_path = potential_pdfs[0]

        if vlm_enabled and pdf_path.exists():
            print(f"🎨 发现关联 PDF: {pdf_path.name}，启动多模态解析...")
            vlm = VLMDescriber(timeout=60)
            # 这里会自动入库到 LanceDB
            await ingest_pdf_multimodal(str(pdf_path), store, vlm_describer=vlm)

        # 手动释放
        embedding_manager.unload()

        t1 = time.time()
        print(f"🎉 增量索引全部完成! 总耗时: {t1 - t0:.2f} 秒。")

        if lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    except Exception as e:
        error_msg = str(e)
        print(f"❌ 增量入库异常: {error_msg}")
        send_feishu_alert(filepath.name, error_msg)
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except: pass
        raise

async def main():
    parser = argparse.ArgumentParser(description="增量索引指定文档")
    parser.add_argument("--file", required=True, type=str, help="要索引的 Markdown 文件路径")
    parser.add_argument("--prefix", type=str, default="vault:", help="文档前缀")
    parser.add_argument("--no-vlm", action="store_true", help="禁用多模态解析 (加速批量入库)")
    args = parser.parse_args()

    filepath = Path(args.file).expanduser().resolve()
    if not filepath.exists() or not filepath.is_file():
        print(f"❌ 文件不存在: {filepath}")
        sys.exit(1)

    await ingest_core(filepath, prefix=args.prefix, vlm_enabled=not args.no_vlm)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
