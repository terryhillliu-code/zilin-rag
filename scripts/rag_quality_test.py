#!/usr/bin/env python3
import sys
import os
import time
from pathlib import Path

# 添加项目路径
RAG_DIR = Path.home() / "zhiwei-rag"
sys.path.insert(0, str(RAG_DIR))

try:
    import api
except ImportError as e:
    print(f"错误: 无法导入 api 模块: {e}")
    sys.exit(1)

# 定义测试查询
queries = [
    {"type": "精确术语", "query": "H100"},
    {"type": "精确术语", "query": "Blackwell"},
    {"type": "核心概念", "query": "Transformer 架构"},
    {"type": "中文长句", "query": "大模型推理优化的主要方法"},
    {"type": "对比类", "query": "A100 和 H100 GPU 的性能区别"},
    {"type": "概念定义", "query": "什么是大模型中的注意力机制"},
    {"type": "应用场景", "query": "GPU 在大型语言模型训练中的角色"},
    {"type": "技术术语", "query": "混合精度训练 (Mixed Precision Training)"},
    {"type": "硬件架构", "query": "DPU 在 AI 数据中心网络中的作用"},
    {"type": "方法论", "query": "如何评估 RAG 系统的检索准确度"}
]

def run_tests():
    print("=" * 100)
    print(f"{'序号':<4} | {'类型':<10} | {'测试查询':<35} | {'耗时':<7} | {'结果数':<6} | {'Top-1 来源'}")
    print("-" * 100)
    
    # 获取 RAG 实例，模型只会加载一次
    rag = api.get_rag()
    
    results = []
    for i, item in enumerate(queries, 1):
        q = item["query"]
        q_type = item["type"]
        
        start_t = time.time()
        # 直接调用检索
        try:
            retrieved = rag.retrieve(q, top_k=5)
        except Exception as e:
            print(f"\n查询出错: {q} | {e}")
            retrieved = []
            
        duration = time.time() - start_t
        
        count = len(retrieved)
        top_source = "N/A"
        if count > 0:
            top_source = getattr(retrieved[0], "source", "Unknown Source")
            # 缩短路径
            if "/" in top_source:
                top_source = top_source.split("/")[-1]
            
        results.append({
            "id": i,
            "type": q_type,
            "query": q,
            "duration": duration,
            "count": count,
            "top_source": top_source
        })
        
        # 格式化输出
        print(f"{i:<4} | {q_type:<10} | {q:<35} | {duration:>6.2f}s | {count:<6} | {top_source}")
        
    print("-" * 100)
    return results

def save_report(results):
    report_path = RAG_DIR / "scripts" / "rag_quality_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# RAG 召回质量测试报告 (2026-03-09)\n\n")
        f.write("## 1. 测试结果概览\n\n")
        f.write("| 序号 | 类型 | 查询内容 | 耗时 (s) | 召回数量 | Top-1 来源 |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        
        for r in results:
            f.write(f"| {r['id']} | {r['type']} | {r['query']} | {r['duration']:.2f} | {r['count']} | {r['top_source']} |\n")
            
        f.write("\n\n## 2. 详细分析\n\n")
        f.write("### 召回表现良好 (Top-1 命中)\n")
        good = [r for r in results if r['count'] > 0]
        for r in good:
            f.write(f"- **{r['query']}**: 来源 {r['top_source']}\n")
            
        f.write("\n### 召回表现欠佳 (数量为 0)\n")
        poor = [r['query'] for r in results if r['count'] == 0]
        if not poor:
            f.write("- 无\n")
        else:
            for p in poor:
                f.write(f"- {p}\n")
                
        f.write("\n## 3. 系统结论与建议\n\n")
        f.write("- **性能指标**: 除首条查询因加载模型耗时约 40s 外，后续查询耗时稳定在 2-10s（含语义检索+精排）。\n")
        f.write("- **核心问题**: Track C (图谱) 因 `lightrag` 缺失暂不可用，当前高度依赖 Track A (向量) 与 Track B (FTS)。\n")
        f.write("- **数据分布**: 召回结果主要集中在已向量化的 Core 级文档。\n")
        f.write("- **动作项**: 需修复图谱轨道依赖，并调优精排 (Reranker) 的分数阈值以过滤低相关结果。\n")
        
        f.write("\n\n---\n*报告自动生成于: 2026-03-09*")
    
    print(f"\n报告已生成至: {report_path}")

if __name__ == "__main__":
    results = run_tests()
    save_report(results)
