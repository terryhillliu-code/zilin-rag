"""三轨检索测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_vector_track():
    """测试向量检索轨道"""
    from retrieve.vector_track import VectorTrack
    from retrieve.embedding_manager import EmbeddingManager
    
    manager = EmbeddingManager(device='mps', idle_timeout=60)
    track = VectorTrack(embedding_manager=manager)
    
    print("\n--- [向量轨道] 测试 ---")
    results = track.search("人工智能深度学习", top_k=5)
    
    print(f"返回 {len(results)} 条结果")
    if results:
        r = results[0]
        print(f"  最相关来源: {r.source}")
        print(f"  分数: {r.score:.4f}")
        print(f"  文本片段: {r.text[:100]}...")
    
    manager.unload()
    print("✅ 向量轨道测试通过")


def test_fts_track():
    """测试 FTS 轨道（LanceDB FTS）"""
    from retrieve.vector_track import VectorTrack

    track = VectorTrack()

    print("\n--- [FTS 轨道] 测试 (LanceDB) ---")
    results = track.search_fts("机器学习", top_k=5)

    print(f"返回 {len(results)} 条结果")
    if results:
        r = results[0]
        print(f"  来源: {r.source[:50]}...")
        print(f"  分数: {r.score:.4f}")
    else:
        print("  (无结果)")

    print("✅ FTS 轨道测试通过")


def test_hybrid_retriever():
    """测试混合检索器"""
    from retrieve.hybrid_retriever import HybridRetriever, HybridConfig
    from retrieve.embedding_manager import EmbeddingManager
    
    manager = EmbeddingManager(device='mps', idle_timeout=60)
    
    config = HybridConfig(
        vector_top_k=10,
        fts_top_k=5,
        vector_weight=0.5,
        fts_weight=0.3
    )
    
    retriever = HybridRetriever(
        config=config,
        embedding_manager=manager
    )
    
    print("\n--- [混合检索] 测试 ---")
    query = "深度学习神经网络"
    print(f"查询: '{query}'")
    results = retriever.search(query, top_k=5)
    
    print(f"最终返回 {len(results)} 条结果")
    for i, r in enumerate(results):
        print(f"  {i+1}. [{r.track}] score={r.score:.4f}")
        print(f"      source: {r.source}")
        print(f"      text sample: {r.text[:80]}...")
    
    manager.unload()
    print("✅ 混合检索测试通过")


if __name__ == '__main__':
    print("=" * 60)
    print("三轨检索详细测试流程")
    print("=" * 60)
    
    test_vector_track()
    test_fts_track()
    test_hybrid_retriever()
    
    print("\n" + "=" * 60)
    print("✅ 所有检索组件基本测试完成")
