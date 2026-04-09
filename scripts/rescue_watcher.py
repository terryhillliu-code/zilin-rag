#!/usr/bin/env python3
"""
MinerU 输出目录 Watchdog
- 扫描 mineru_output 目录
- 检查已解析的 .md 文件是否已入库
- 自动触发 rescue 入库逻辑
- 清理已处理的临时目录
"""
import os
import sys
import argparse
import shutil
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.mineru_extractor import MinerUExtractor
from ingest.semantic_splitter import SemanticSplitter
from ingest.lance_store import LanceStore, Document
from retrieve.embedding_manager import EmbeddingManager


class MinerURescuer:
    """MinerU 滞留文档补录器"""

    OUTPUT_DIR = Path.home() / "zhiwei-rag" / "mineru_output"
    DB_PATH = Path.home() / "zhiwei-rag" / "data" / "lance_db"

    def __init__(self, dry_run: bool = False):
        """
        Args:
            dry_run: 仅分析不执行入库
        """
        self.dry_run = dry_run
        self.store = None
        self.splitter = None
        self.embedding_manager = None

        if not dry_run:
            self._init_components()

    def _init_components(self):
        """初始化入库组件"""
        import yaml

        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.embedding_manager = EmbeddingManager(
            model_name=config['embedding']['model_name'],
            device=config['embedding']['device'],
            idle_timeout=config['embedding']['idle_timeout']
        )

        self.store = LanceStore(
            db_path=config['paths']['lance_db'],
            embedding_manager=self.embedding_manager
        )

        self.splitter = SemanticSplitter(
            max_chunk_tokens=config['splitter']['max_chunk_tokens'],
            min_chunk_chars=config['splitter']['min_chunk_chars'],
            fallback_to_paragraph=config['splitter']['fallback_to_paragraph']
        )

    def scan_output_dir(self) -> List[dict]:
        """
        扫描 mineru_output 目录

        Returns:
            滞留文档列表，每个元素包含:
            - dir_name: 目录名
            - md_path: Markdown 文件路径
            - pdf_name: PDF 文件名
            - timestamp: 创建时间戳
            - status: pending/processed/error
        """
        results = []

        if not self.OUTPUT_DIR.exists():
            print(f"[Rescuer] 输出目录不存在: {self.OUTPUT_DIR}")
            return results

        for subdir in self.OUTPUT_DIR.iterdir():
            if not subdir.is_dir():
                continue

            # 解析目录名：{pdf_name}_{timestamp}
            dir_name = subdir.name
            timestamp = None

            # 提取时间戳
            import re
            match = re.search(r'_(\d{10})$', dir_name)
            if match:
                timestamp = int(match.group(1))
                pdf_name = dir_name[:match.start()]
            else:
                pdf_name = dir_name

            # 查找 .md 文件
            md_path = self._find_markdown(subdir)

            if md_path:
                # 检查是否已入库
                status = self._check_status(md_path)
            else:
                status = "pending"  # 还在解析中

            results.append({
                "dir_name": dir_name,
                "md_path": md_path,
                "pdf_name": pdf_name,
                "timestamp": timestamp,
                "status": status,
                "dir_path": subdir
            })

        return results

    def _find_markdown(self, subdir: Path) -> Optional[Path]:
        """查找目录中的 Markdown 文件"""
        # MinerU 输出结构: subdir/{pdf_name}/auto/{pdf_name}.md
        for child in subdir.iterdir():
            if child.is_dir():
                auto_dir = child / "auto"
                if auto_dir.exists():
                    md_files = list(auto_dir.glob("*.md"))
                    if md_files:
                        return md_files[0]

                # 也检查 hybrid_auto
                hybrid_dir = child / "hybrid_auto"
                if hybrid_dir.exists():
                    md_files = list(hybrid_dir.glob("*.md"))
                    if md_files:
                        return md_files[0]

        return None

    def _check_status(self, md_path: Path) -> str:
        """
        检查文档是否已在向量库

        Returns:
            'indexed' / 'pending'
        """
        if self.dry_run or self.store is None:
            return "pending"

        # 查询向量库
        source = str(md_path)
        try:
            # 转义单引号，防止 SQL 截断
            safe_name = md_path.name.replace("'", "''")
            # 使用 search 检查是否存在
            results = self.store.table.search("").where(f"source LIKE '%{safe_name}%'").limit(1).to_list()
            if results:
                return "indexed"
        except Exception:
            pass

        return "pending"

    def rescue(self, items: List[dict]) -> dict:
        """
        补录滞留文档

        Args:
            items: 待补录文档列表

        Returns:
            统计结果
        """
        stats = {
            "total": len(items),
            "indexed": 0,
            "failed": 0,
            "skipped": 0,
            "cleaned": 0
        }

        for item in items:
            if item["status"] == "indexed":
                stats["skipped"] += 1
                continue

            if not item["md_path"]:
                stats["skipped"] += 1
                continue

            if self.dry_run:
                print(f"[Rescuer] [DRY-RUN] 将入库: {item['md_path']}")
                stats["indexed"] += 1
                continue

            # 实际入库
            try:
                count = self._ingest_markdown(item["md_path"])
                stats["indexed"] += count

                # 清理目录
                self._cleanup_dir(item["dir_path"])
                stats["cleaned"] += 1

            except Exception as e:
                print(f"[Rescuer] 入库失败: {item['md_path']} - {e}")
                stats["failed"] += 1

        return stats

    def _ingest_markdown(self, md_path: Path) -> int:
        """入库单个 Markdown 文件"""
        # 使用 split_file 分块
        chunks = list(self.splitter.split_file(md_path, extra_metadata={"category": "PDF研报"}))

        if not chunks:
            return 0

        # 获取向量
        texts = [c.text for c in chunks]
        embeddings = self.embedding_manager.encode(texts)

        # 转换为 Document
        docs = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            doc = Document(
                id=f"mineru:{md_path.stem}#{i}",
                text=chunk.text,
                raw_text=chunk.raw_text,
                source=chunk.source,
                filename=chunk.filename,
                h1=chunk.h1,
                h2=chunk.h2,
                category="PDF研报",
                tags="mineru",
                char_count=chunk.char_count,
                vector=vector.tolist()
            )
            docs.append(doc)

        # 写入向量库
        self.store.add_documents(docs)

        print(f"[Rescuer] 入库: {md_path.name} - {len(docs)} 条")
        return len(docs)

    def _cleanup_dir(self, dir_path: Path):
        """清理已处理的目录"""
        try:
            shutil.rmtree(dir_path)
            print(f"[Rescuer] 清理目录: {dir_path.name}")
        except Exception as e:
            print(f"[Rescuer] 清理失败: {dir_path.name} - {e}")

    def run(self) -> dict:
        """执行完整补录流程"""
        print("=" * 60)
        print("MinerU 滞留文档补录")
        print("=" * 60)
        print(f"模式: {'DRY-RUN' if self.dry_run else '执行'}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 扫描
        print("\n[1] 扫描 mineru_output...")
        items = self.scan_output_dir()

        # 分类统计
        pending = [i for i in items if i["status"] == "pending"]
        indexed = [i for i in items if i["status"] == "indexed"]
        no_md = [i for i in items if not i["md_path"]]

        print(f"    发现 {len(items)} 个目录")
        print(f"    - 待入库: {len(pending)}")
        print(f"    - 已入库: {len(indexed)}")
        print(f"    - 无 md 文件: {len(no_md)}")

        # 补录
        if pending:
            print("\n[2] 补录滞留文档...")
            stats = self.rescue(pending)

            print("\n" + "=" * 60)
            print("补录完成")
            print(f"  - 入库: {stats['indexed']} 条")
            print(f"  - 失败: {stats['failed']} 个")
            print(f"  - 清理目录: {stats['cleaned']} 个")
            print("=" * 60)

            return stats
        else:
            print("\n无待入库文档")
            return {"total": 0, "indexed": 0, "failed": 0, "skipped": 0, "cleaned": 0}


def main():
    parser = argparse.ArgumentParser(description="MinerU 滞留文档补录")
    parser.add_argument("--dry-run", action="store_true", help="仅分析不执行")
    args = parser.parse_args()

    rescuer = MinerURescuer(dry_run=args.dry_run)
    rescuer.run()


if __name__ == "__main__":
    main()