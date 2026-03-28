#!/usr/bin/env python3
"""
EPUB 解析与向量化模块
- 提取 EPUB 文字内容
- 分块并向量化入库
"""
import re
import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from ebooklib import epub
from bs4 import BeautifulSoup

from ingest.semantic_splitter import SemanticSplitter, Chunk
from ingest.lance_store import LanceStore
from retrieve.embedding_manager import EmbeddingManager


@dataclass
class EPUBContent:
    """EPUB 提取结果"""
    title: str
    text: str
    chapters: List[str]


def extract_epub_content(epub_path: str) -> Optional[EPUBContent]:
    """
    从 EPUB 文件提取文字内容

    Args:
        epub_path: EPUB 文件路径

    Returns:
        EPUBContent 或 None（失败时）
    """
    try:
        book = epub.read_epub(epub_path)

        # 获取标题
        title = book.get_metadata('DC', 'title')
        title = title[0][0] if title else Path(epub_path).stem

        # 提取所有 HTML 内容
        chapters = []
        full_text = []

        for item in book.get_items():
            if item.get_type() == 9:  # ITEM_DOCUMENT
                content = item.get_content()
                soup = BeautifulSoup(content, 'html.parser')

                # 移除脚本和样式
                for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                    tag.decompose()

                # 提取文字
                text = soup.get_text(separator='\n')

                # 清理
                lines = []
                for line in text.split('\n'):
                    line = line.strip()
                    if line and len(line) > 1:
                        lines.append(line)

                chapter_text = '\n'.join(lines)
                if chapter_text.strip():
                    chapters.append(chapter_text)
                    full_text.append(chapter_text)

        return EPUBContent(
            title=title,
            text='\n\n'.join(full_text),
            chapters=chapters
        )

    except Exception as e:
        print(f"   ❌ EPUB 解析失败: {e}")
        return None


def ingest_epub(
    epub_path: str,
    store: LanceStore,
    embedding_manager: EmbeddingManager,
    source_prefix: str = "epub:"
) -> dict:
    """
    处理单个 EPUB 文件并向量化入库

    Args:
        epub_path: EPUB 文件路径
        store: LanceDB 存储
        embedding_manager: Embedding 管理器
        source_prefix: 来源前缀

    Returns:
        {'total': int, 'success': bool}
    """
    print(f"[EPUB] 处理: {Path(epub_path).name}")

    # 提取内容
    content = extract_epub_content(epub_path)
    if not content or not content.text.strip():
        print(f"   ⚠️ 无法提取内容或内容为空")
        return {'total': 0, 'success': False}

    print(f"   标题: {content.title}")
    print(f"   章节: {len(content.chapters)}")
    print(f"   文字: {len(content.text):,} 字符")

    # 分块
    splitter = SemanticSplitter(
        max_chunk_tokens=480,
        min_chunk_chars=100,
        fallback_to_paragraph=True
    )

    # 将全文转为临时文件供 splitter 处理
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(f"# {content.title}\n\n")
        for i, chapter in enumerate(content.chapters):
            f.write(f"## 章节 {i+1}\n\n{chapter}\n\n")
        temp_path = f.name

    try:
        chunks = splitter.split_file(Path(temp_path))
    finally:
        Path(temp_path).unlink()

    if not chunks:
        print(f"   ⚠️ 分块失败")
        return {'total': 0, 'success': False}

    print(f"   分块: {len(chunks)} chunks")

    # 向量化
    texts = [c.text for c in chunks]
    embeddings = embedding_manager.encode(texts)

    # 构建文档
    from ingest.ingest_all import chunks_to_documents
    docs = chunks_to_documents(chunks, embeddings, source_prefix=source_prefix)

    # 删除旧数据
    store.delete_by_source(epub_path)

    # 入库
    store.add_documents(docs)

    print(f"   ✅ 入库完成: {len(docs)} 条文档")

    return {'total': len(docs), 'success': True}


async def bulk_ingest_epub(file_list_path: str):
    """
    批量处理 EPUB 文件

    Args:
        file_list_path: 文件列表路径
    """
    import sqlite3

    with open(file_list_path, 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    print(f"📚 EPUB 批量入库: {len(files)} 个文件")

    # 初始化
    store = LanceStore()
    emb = EmbeddingManager(model_name='BAAI/bge-large-zh-v1.5', device='mps')
    emb.preload()

    # klib.db
    klib_path = Path.home() / "Documents/Library/klib.db"

    success = 0
    fail = 0

    for i, epub_path in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {Path(epub_path).name}")

        if not Path(epub_path).exists():
            print(f"   ❌ 文件不存在")
            fail += 1
            continue

        result = ingest_epub(epub_path, store, emb)

        if result['success']:
            success += 1
            # 更新 klib.db
            if klib_path.exists():
                try:
                    conn = sqlite3.connect(str(klib_path))
                    conn.execute(
                        "UPDATE books SET vectorized = 1 WHERE file_path = ?",
                        (epub_path,)
                    )
                    conn.commit()
                    conn.close()
                except:
                    pass
        else:
            fail += 1

    print(f"\n📊 完成! 成功: {success}, 失败: {fail}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EPUB 向量化入库")
    parser.add_argument("file_list", help="EPUB 文件列表")
    args = parser.parse_args()

    import asyncio
    asyncio.run(bulk_ingest_epub(args.file_list))