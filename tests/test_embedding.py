"""Embedding 和 LanceDB 测试"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_embedding_manager():
    """测试 Embedding 管理器"""
    from retrieve.embedding_manager import EmbeddingManager
    
    manager = EmbeddingManager(
        model_name='BAAI/bge-large-zh-v1.5',
        device='mps',
        idle_timeout=10  # 测试用短超时
    )
    
    # 测试首次加载
    print("测试首次加载...")
    start = time.time()
    vec1 = manager.encode(["这是一个测试句子"])
    load_time = time.time() - start
    print(f"  首次加载耗时: {load_time:.1f}s")
    print(f"  向量维度: {vec1.shape}")
    
    # 测试后续编码（应该快很多）
    start = time.time()
    vec2 = manager.encode(["另一个测试", "第三个测试"])
    encode_time = time.time() - start
    print(f"  后续编码耗时: {encode_time:.3f}s")
    print(f"  向量 shape: {vec2.shape}")
    
    # 验证向量归一化
    import numpy as np
    norm = np.linalg.norm(vec1[0])
    print(f"  向量模长: {norm:.4f} (应接近 1.0)")
    
    # 手动卸载
    manager.unload()
    assert not manager.is_loaded
    print("✅ Embedding 管理器测试通过")


def test_lance_store():
    """测试 LanceDB 存储"""
    import tempfile
    import numpy as np
    from ingest.lance_store import LanceStore, Document
    
    # 使用临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceStore(db_path=tmpdir)
        store.create_table(dimension=4)  # 小维度便于测试
        
        # 添加文档
        docs = [
            Document(
                id="doc1",
                text="关于深度学习的介绍",
                raw_text="深度学习介绍",
                source="/test/doc1.md",
                filename="doc1",
                h1="深度学习",
                h2="介绍",
                category="AI",
                tags="深度学习, 神经网络",
                char_count=10,
                vector=[0.1, 0.2, 0.3, 0.4]
            ),
            Document(
                id="doc2",
                text="关于机器学习的基础",
                raw_text="机器学习基础",
                source="/test/doc2.md",
                filename="doc2",
                h1="机器学习",
                h2="基础",
                category="AI",
                tags="机器学习",
                char_count=8,
                vector=[0.2, 0.3, 0.4, 0.5]
            ),
        ]
        store.add_documents(docs)
        
        # 验证数量
        assert store.count() == 2, f"期望 2 条，实际 {store.count()}"
        
        # 测试检索
        query_vec = np.array([0.1, 0.2, 0.3, 0.4])
        results = store.search(query_vec, top_k=2)
        assert len(results) == 2
        # 最相似的应该是 doc1
        assert results[0]['id'] == 'doc1'
        
        # 测试过滤
        results = store.search(query_vec, top_k=2, filter_sql="category = 'AI'")
        assert len(results) == 2
        
        print("✅ LanceDB 存储测试通过")


def test_end_to_end():
    """端到端测试：切分 -> 编码 -> 存储 -> 检索"""
    import tempfile
    from pathlib import Path
    from ingest.semantic_splitter import SemanticSplitter
    from ingest.lance_store import LanceStore, Document
    from retrieve.embedding_manager import EmbeddingManager
    
    # 创建测试文件
    test_content = """---
category: 测试分类
tags:
  - 标签1
  - 标签2
---

# 测试文档

## 第一节

这是第一节的内容，讲述了一些关于人工智能的基础知识。

## 第二节

这是第二节，继续深入探讨神经网络的原理。
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试文件
        test_file = Path(tmpdir) / "test.md"
        test_file.write_text(test_content)
        
        # 切分
        splitter = SemanticSplitter()
        chunks = splitter.split_file(test_file)
        print(f"  切分得到 {len(chunks)} 个 chunk")
        
        # 编码
        manager = EmbeddingManager(device='mps', idle_timeout=10)
        texts = [c.text for c in chunks]
        embeddings = manager.encode(texts)
        print(f"  编码得到 {embeddings.shape} 向量")
        
        # 存储
        store = LanceStore(db_path=tmpdir, embedding_manager=manager)
        store.create_table(dimension=embeddings.shape[1])
        
        docs = []
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            docs.append(Document(
                id=f"test#{i}",
                text=chunk.text,
                raw_text=chunk.raw_text,
                source=str(test_file),
                filename=chunk.filename,
                h1=chunk.h1,
                h2=chunk.h2,
                category=chunk.metadata.get('category', ''),
                tags=str(chunk.metadata.get('tags', [])),
                char_count=chunk.char_count,
                vector=vec.tolist()
            ))
        store.add_documents(docs)
        print(f"  存储 {store.count()} 条文档")
        
        # 检索
        results = store.search_text("人工智能基础", top_k=2)
        print(f"  检索 '人工智能基础' 得到 {len(results)} 条结果")
        if results:
            print(f"    最相关: {results[0]['filename']} - {results[0]['h2']}")
            print(f"    距离: {results[0]['_distance']:.4f}")
        
        # 清理
        manager.unload()
        
        print("✅ 端到端测试通过")


if __name__ == '__main__':
    print("=" * 50)
    print("Embedding + LanceDB 测试")
    print("=" * 50)
    test_embedding_manager()
    print()
    test_lance_store()
    print()
    test_end_to_end()
    print("=" * 50)
    print("✅ 所有测试通过")
