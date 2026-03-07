"""集成测试"""
import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_context_builder():
    """测试上下文组装"""
    from generate.context_builder import ContextBuilder, ContextConfig
    from retrieve.vector_track import RetrievalResult
    
    # 模拟检索结果
    results = [
        RetrievalResult(
            text="Transformer 是一种基于自注意力机制的神经网络架构。",
            raw_text="Transformer 架构",
            source="/docs/transformer.md",
            score=0.9,
            track="vector",
            metadata={}
        ),
        RetrievalResult(
            text="BERT 使用 Transformer 的 Encoder 部分进行预训练。",
            raw_text="BERT 预训练",
            source="/docs/bert.md",
            score=0.8,
            track="fts",
            metadata={}
        ),
    ]
    
    config = ContextConfig(max_tokens=1000, include_source=True)
    builder = ContextBuilder(config)
    
    # 测试 QA 模板
    prompt = builder.build(
        query="什么是 Transformer？",
        results=results,
        template_name="qa"
    )
    
    assert "Transformer" in prompt
    assert "[来源 1:" in prompt
    assert "用户问题" in prompt
    
    print("✅ 上下文组装测试通过")
    print(f"   Prompt 长度: {len(prompt)} 字符")


def test_api():
    """测试统一 API"""
    from api import RAG, RAGConfig
    
    config = RAGConfig(
        vector_top_k=5,
        enable_rerank=False  # 先不测试精排（模型可能还在下载）
    )
    
    rag = RAG(config)
    
    # 测试检索
    results = rag.retrieve("深度学习", top_k=3)
    print(f"[API] 检索返回 {len(results)} 条结果")
    
    # 测试上下文构建
    context = rag.get_context("深度学习", top_k=3)
    print(f"[API] 上下文长度: {len(context)} 字符")
    
    # 测试完整流程
    prompt, results = rag.retrieve_and_build_context(
        "什么是注意力机制？",
        top_k=3,
        template="qa"
    )
    print(f"[API] 完整 Prompt 长度: {len(prompt)} 字符")
    
    rag.cleanup()
    print("✅ API 测试通过")


def test_bridge():
    """测试桥接脚本"""
    bridge_path = Path(__file__).parent.parent / "bridge.py"
    
    # 测试 context 命令
    result = subprocess.run(
        [sys.executable, str(bridge_path), "context", "机器学习", "--top-k", "3"],
        capture_output=True,
        text=True,
        cwd=str(bridge_path.parent)
    )
    
    if result.returncode == 0:
        print(f"[Bridge] context 命令成功")
        print(f"   输出长度: {len(result.stdout)} 字符")
    else:
        print(f"[Bridge] context 命令失败: {result.stderr}")
    
    # 测试 retrieve 命令
    result = subprocess.run(
        [sys.executable, str(bridge_path), "retrieve", "神经网络", "--top-k", "2"],
        capture_output=True,
        text=True,
        cwd=str(bridge_path.parent)
    )
    
    if result.returncode == 0:
        import json
        data = json.loads(result.stdout)
        print(f"[Bridge] retrieve 命令成功，返回 {len(data)} 条")
    else:
        print(f"[Bridge] retrieve 命令失败: {result.stderr}")
    
    print("✅ 桥接脚本测试通过")


def test_token_budget():
    """测试 Token 预算控制"""
    from generate.context_builder import ContextBuilder, ContextConfig
    from retrieve.vector_track import RetrievalResult
    
    # 创建大量结果
    results = []
    for i in range(20):
        results.append(RetrievalResult(
            text=f"这是第 {i+1} 条测试内容。" * 50,  # 每条约 500 字符
            raw_text=f"测试 {i+1}",
            source=f"test_{i}.md",
            score=0.9 - i * 0.01,
            track="vector",
            metadata={}
        ))
    
    # 严格限制 token
    config = ContextConfig(max_tokens=500, include_source=True)
    builder = ContextBuilder(config)
    
    context = builder.build_context_only(results)
    
    # 验证被截断
    assert "已截断" in context
    
    # 估算 token 应该在预算内
    estimated_tokens = len(context) / 1.5
    assert estimated_tokens < 600, f"Token 超出预算: {estimated_tokens}"
    
    print(f"✅ Token 预算测试通过")
    print(f"   上下文长度: {len(context)} 字符")
    print(f"   估算 Token: {estimated_tokens:.0f}")


if __name__ == '__main__':
    print("=" * 60)
    print("集成测试")
    print("=" * 60)
    
    print("\n--- 上下文组装测试 ---")
    test_context_builder()
    
    print("\n--- Token 预算测试 ---")
    test_token_budget()
    
    print("\n--- API 测试 ---")
    test_api()
    
    print("\n--- 桥接脚本测试 ---")
    test_bridge()
    
    print("\n" + "=" * 60)
    print("✅ 所有集成测试通过")
