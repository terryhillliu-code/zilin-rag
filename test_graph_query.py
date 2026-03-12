import time
import sys
import os

# 确保能导入 zhiwei-rag 中的模块
sys.path.append(os.path.expanduser('~/zhiwei-rag'))

try:
    from retrieve.graph_track import GraphTrack
    print("🚀 初始化 GraphTrack...")
    g = GraphTrack()
    
    queries = ["GPU", "液冷", "CXL", "服务器"]
    print(f"\n开始基准测试 (关键词: {queries})")
    print("-" * 50)
    
    for query in queries:
        start_time = time.time()
        try:
            print(f"正在查询: [{query}] ...", end="", flush=True)
            result = g.search(query, top_k=3)
            elapsed = (time.time() - start_time) * 1000
            print(f" 完成! (耗时: {elapsed:.2f}ms, 结果数: {len(result)})")
            if result:
                for i, res in enumerate(result):
                    text = getattr(res, 'text', getattr(res, 'content', str(res)))
                    print(f"  [{i+1}] {str(text)[:80]}...")
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            print(f" ❌ 失败! (耗时: {elapsed:.2f}ms) 错误: {e}")
            
    print("-" * 50)
    print("✅ 基准测试执行完毕")

except Exception as e:
    print(f"❌ 初始化失败: {e}")
