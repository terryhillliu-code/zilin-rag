"""
全量索引入口
将 Obsidian Vault 和研报库索引到 LanceDB
"""
import sys
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.semantic_splitter import SemanticSplitter, Chunk
from ingest.lance_store import LanceStore, Document
from retrieve.embedding_manager import EmbeddingManager


def chunks_to_documents(
    chunks: list[Chunk],
    embeddings,
    source_prefix: str = ""
) -> list[Document]:
    """将 Chunk 转换为 Document"""
    docs = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        doc_id = f"{source_prefix}{chunk.source}#{i}"
        
        # 提取元数据
        category = chunk.metadata.get('category', '')
        tags = chunk.metadata.get('tags', [])
        if isinstance(tags, list):
            tags = ', '.join(str(t) for t in tags)
        
        doc = Document(
            id=doc_id,
            text=chunk.text,
            raw_text=chunk.raw_text,
            source=chunk.source,
            filename=chunk.filename,
            h1=chunk.h1,
            h2=chunk.h2,
            category=str(category),
            tags=str(tags),
            char_count=chunk.char_count,
            vector=vector.tolist()
        )
        docs.append(doc)
    
    return docs


def index_directory(
    dir_path: Path,
    store: LanceStore,
    embedding_manager: EmbeddingManager,
    splitter: SemanticSplitter,
    pattern: str = "**/*.md",
    batch_size: int = 50,
    source_prefix: str = "",
    skip_dirs: Optional[list[str]] = None
) -> int:
    """
    索引目录

    Args:
        skip_dirs: DKI 隔离目录列表（从 config.yaml 读取）

    Returns:
        索引的 chunk 数量
    """
    # 收集所有 chunk
    print(f"[Ingest] 扫描目录: {dir_path}")
    if skip_dirs:
        print(f"[Ingest] DKI 隔离目录: {skip_dirs}")
    all_chunks = list(splitter.split_directory(dir_path, pattern=pattern, skip_dirs=skip_dirs))
    print(f"[Ingest] 共 {len(all_chunks)} 个 chunk")
    
    if not all_chunks:
        return 0
    
    # 分批处理
    total_indexed = 0
    
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="索引中"):
        batch_chunks = all_chunks[i:i + batch_size]
        
        # 提取文本进行编码
        texts = [c.text for c in batch_chunks]
        embeddings = embedding_manager.encode(texts)
        
        # 转换并写入
        docs = chunks_to_documents(batch_chunks, embeddings, source_prefix)
        store.add_documents(docs)
        
        total_indexed += len(docs)
    
    return total_indexed


def main():
    """主函数"""
    import yaml
    
    # 加载配置
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 初始化组件
    print("=" * 60)
    print("zhiwei-rag 全量索引")
    print("=" * 60)
    
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
    
    # 清空旧数据（全量重建）
    store.clear()
    
    # 预加载模型并创建表
    embedding_manager.preload()
    store.create_table(dimension=embedding_manager.dimension)
    
    start_time = time.time()
    total_chunks = 0
    
    # 索引 Obsidian Vault
    vault_path = Path(config['paths']['obsidian_vault']).expanduser()
    if vault_path.exists():
        print(f"\n[1/2] 索引 Obsidian Vault: {vault_path}")
        # 从配置读取 DKI 隔离目录
        skip_dirs = config.get('dki', {}).get('skip_dirs', None)
        count = index_directory(
            vault_path, store, embedding_manager, splitter,
            source_prefix="vault:", skip_dirs=skip_dirs
        )
        total_chunks += count
        print(f"      完成: {count} 个 chunk")
    else:
        print(f"\n[1/2] 跳过 Obsidian Vault（目录不存在）: {vault_path}")
    
    # 索引研报库（如果有 Markdown 文件）
    research_path = Path(config['paths']['research_library']).expanduser()
    if research_path.exists():
        print(f"\n[2/2] 索引研报库: {research_path}")
        skip_dirs = config.get('dki', {}).get('skip_dirs', None)
        count = index_directory(
            research_path, store, embedding_manager, splitter,
            source_prefix="research:", skip_dirs=skip_dirs
        )
        total_chunks += count
        print(f"      完成: {count} 个 chunk")
    else:
        print(f"\n[2/2] 跳过研报库（目录不存在）: {research_path}")
    
    elapsed = time.time() - start_time
    
    # 手动释放模型
    embedding_manager.unload()
    
    print("\n" + "=" * 60)
    print(f"索引完成!")
    print(f"  总 chunk 数: {total_chunks}")
    print(f"  数据库大小: {store.count()} 条")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  数据位置: {store.db_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
