"""系统集成验证测试"""
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_scheduler_bridge():
    """测试 scheduler 的 RAG 桥接"""
    bridge_path = Path.home() / "zhiwei-scheduler" / "rag_bridge.py"
    
    if not bridge_path.exists():
        print("⚠️ zhiwei-scheduler/rag_bridge.py 不存在")
        return False
    
    # 尝试导入
    sys.path.insert(0, str(bridge_path.parent))
    try:
        import rag_bridge
        
        # 测试可用性检查
        available = rag_bridge.is_available()
        print(f"[Scheduler Bridge] 可用: {available}")
        
        if available:
            # 测试检索
            context = rag_bridge.get_context("机器学习", top_k=3)
            print(f"[Scheduler Bridge] 上下文长度: {len(context)} 字符")
            
            if len(context) > 50:
                print("✅ Scheduler 桥接测试通过")
                return True
            else:
                print("⚠️ Scheduler 桥接返回内容过短")
                return False
        else:
            print("⚠️ zhiwei-rag 不可用")
            return False
            
    except Exception as e:
        print(f"❌ Scheduler 桥接测试失败: {e}")
        return False


def test_bot_bridge():
    """测试 bot 的 RAG 桥接"""
    bridge_path = Path.home() / "zhiwei-bot" / "rag_bridge.py"
    
    if not bridge_path.exists():
        print("⚠️ zhiwei-bot/rag_bridge.py 不存在")
        return False
    
    sys.path.insert(0, str(bridge_path.parent))
    try:
        # 需要重新导入以避免缓存
        import importlib
        if 'rag_bridge' in sys.modules:
            del sys.modules['rag_bridge']
        
        import rag_bridge
        
        available = rag_bridge.is_available()
        print(f"[Bot Bridge] 可用: {available}")
        
        if available:
            context = rag_bridge.get_context("机器学习", top_k=3)
            print(f"[Bot Bridge] 上下文长度: {len(context)} 字符")
            
            if len(context) > 50:
                print("✅ Bot 桥接测试通过")
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ Bot 桥接测试失败: {e}")
        return False


def test_end_to_end():
    """端到端测试：模拟完整流程"""
    print("\n[E2E] 模拟 info_brief RAG 增强...")
    
    from api import get_rag
    
    rag = get_rag()
    
    # 模拟简报场景
    query = "最近人工智能领域有哪些重要进展？"
    
    prompt, results = rag.retrieve_and_build_context(
        query,
        top_k=5,
        template="brief"
    )
    
    print(f"[E2E] 检索到 {len(results)} 条结果")
    print(f"[E2E] Prompt 长度: {len(prompt)} 字符")
    print(f"[E2E] Prompt 预览:\n{prompt[:500]}...")
    
    rag.cleanup()
    
    print("✅ 端到端测试通过")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("系统集成验证")
    print("=" * 60)
    
    results = []
    
    print("\n--- Scheduler 桥接测试 ---")
    results.append(("Scheduler Bridge", test_scheduler_bridge()))
    
    print("\n--- Bot 桥接测试 ---")
    results.append(("Bot Bridge", test_bot_bridge()))
    
    print("\n--- 端到端测试 ---")
    results.append(("E2E", test_end_to_end()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✅ 所有集成测试通过")
    else:
        print("⚠️ 部分测试未通过，请检查")
