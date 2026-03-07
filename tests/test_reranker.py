"""Reranker 精排测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_reranker_basic():
    """测试基础精排功能"""
    from rank.reranker import Reranker
    from retrieve.vector_track import RetrievalResult
    
    # 模拟检索结果
    mock_results = [
        RetrievalResult(
            text="深度学习是机器学习的一个分支，使用多层神经网络进行特征学习。",
            raw_text="深度学习是机器学习的一个分支",
            source="test1.md",
            score=0.8,
            track="vector",
            metadata={}
        ),
        RetrievalResult(
            text="今天天气很好，适合出去散步。",
            raw_text="今天天气很好",
            source="test2.md",
            score=0.7,
            track="vector",
            metadata={}
        ),
        RetrievalResult(
            text="神经网络由多个层组成，包括输入层、隐藏层和输出层。",
            raw_text="神经网络由多个层组成",
            source="test3.md",
            score=0.6,
            track="fts",
            metadata={}
        ),
    ]
    
    reranker = Reranker(score_threshold=0.01)
    
    query = "什么是深度学习和神经网络？"
    results = reranker.rerank(query, mock_results, top_k=3)
    
    print(f"[Reranker] 输入 {len(mock_results)} 条，输出 {len(results)} 条")
    
    for i, r in enumerate(results):
        print(f"  {i+1}. rerank_score={r.rerank_score:.4f} | original={r.original_score:.4f}")
        print(f"      {r.raw_text[:50]}...")
    
    # 验证：关于深度学习和神经网络的应该排在前面
    assert len(results) >= 1
    # 天气相关的应该分数较低或被过滤
    weather_scores = [r.rerank_score for r in results if "天气" in r.raw_text]
    dl_scores = [r.rerank_score for r in results if "深度学习" in r.raw_text or "神经网络" in r.raw_text]
    
    if weather_scores and dl_scores:
        assert max(dl_scores) > max(weather_scores), "深度学习相关内容应该分数更高"
    
    print("✅ 基础精排测试通过")


def test_reranker_with_hybrid():
    """测试精排与混合检索的集成"""
    from retrieve.hybrid_retriever import HybridRetriever, HybridConfig
    from retrieve.embedding_manager import EmbeddingManager
    
    manager = EmbeddingManager(device='mps', idle_timeout=60)
    
    config = HybridConfig(
        vector_top_k=10,
        fts_top_k=5,
        graph_top_k=3,
        enable_rerank=True,
        rerank_top_k=5,
        rerank_threshold=0.05
    )
    
    retriever = HybridRetriever(config=config, embedding_manager=manager)
    
    query = "Transformer 模型的注意力机制原理"
    print(f"\n[集成测试] 查询: '{query}'")
    
    # 带精排的检索
    results_with_rerank = retriever.search(query, top_k=5, use_rerank=True)
    print(f"[带精排] 返回 {len(results_with_rerank)} 条")
    
    for i, r in enumerate(results_with_rerank):
        print(f"  {i+1}. [{r.track}] rerank={r.rerank_score:.4f}")
        print(f"      {r.text[:80]}...")
    
    # 不带精排的检索
    results_without_rerank = retriever.search(query, top_k=5, use_rerank=False)
    print(f"\n[无精排] 返回 {len(results_without_rerank)} 条")
    
    for i, r in enumerate(results_without_rerank):
        print(f"  {i+1}. [{r.track}] score={r.score:.4f}")
        print(f"      {r.text[:80]}...")
    
    manager.unload()
    print("\n✅ 精排集成测试通过")


def test_reranker_memory_release():
    """测试模型内存释放"""
    import gc
    import torch
    from rank.reranker import Reranker
    from retrieve.vector_track import RetrievalResult
    
    mock_results = [
        RetrievalResult(
            text="测试文本",
            raw_text="测试",
            source="test.md",
            score=0.5,
            track="test",
            metadata={}
        )
    ]
    
    reranker = Reranker()
    
    # 执行精排（会加载模型）
    reranker.rerank("测试查询", mock_results, top_k=1)
    
    # 验证模型已释放
    assert reranker._model is None, "模型应该已被释放"
    assert reranker._tokenizer is None, "Tokenizer 应该已被释放"
    
    # 尝试清理 MPS 缓存
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    gc.collect()
    
    print("✅ 内存释放测试通过")


if __name__ == '__main__':
    print("=" * 60)
    print("Reranker 精排测试")
    print("=" * 60)
    
    print("\n--- 基础精排测试 ---")
    test_reranker_basic()
    
    print("\n--- 集成测试 ---")
    test_reranker_with_hybrid()
    
    print("\n--- 内存释放测试 ---")
    test_reranker_memory_release()
    
    print("\n" + "=" * 60)
    print("✅ 所有精排测试通过")
