# ~/zhiwei-rag/eval/retrieval_scorer.py
"""
检索跑分器
- 读取测试集
- 执行检索
- 计算 Hit Rate@K 和 MRR
"""
import json
import time
import sys
import argparse
from pathlib import Path

# 添加 zhiwei-rag 到路径
sys.path.insert(0, str(Path.home() / "zhiwei-rag"))

from api import RAG, RAGConfig


def score_retrieval(dataset_path: str = "golden_20.json", top_k: int = 5):
    """
    执行检索跑分

    Args:
        dataset_path: 测试集文件路径
        top_k: 检索返回数量
    """
    eval_dir = Path(__file__).parent
    full_path = eval_dir / dataset_path

    with open(full_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"加载测试集: {len(dataset)} 条")
    print(f"检索参数: top_k={top_k}")
    print("-" * 50)

    # 初始化 RAG（只需一次，复用 embedding 和 rerank 服务）
    print("初始化 RAG 引擎...")
    print("📊 图谱轨道已启用")
    config = RAGConfig(rerank_top_k=top_k, enable_graph=True)
    rag = RAG(config)

    hits = 0
    mrr_sum = 0.0
    total_time = 0

    for i, item in enumerate(dataset):
        query = item['query']
        expected_id = item['expected_id']
        expected_source = item.get('expected_source', '')

        start = time.time()
        results = rag.retrieve(query, top_k=top_k)
        elapsed = time.time() - start
        total_time += elapsed

        # 计算 Hit Rate - 从 source 字段匹配
        found_sources = []
        for r in results:
            source = getattr(r, 'source', '') or ''
            found_sources.append(source)

        hit = False
        rank = 0

        # 优先用 expected_source 匹配（更宽松）
        if expected_source and expected_source in found_sources:
            hits += 1
            hit = True
            rank = found_sources.index(expected_source) + 1
            mrr_sum += 1.0 / rank
        # 否则尝试用 expected_id 匹配
        elif expected_id:
            for idx, src in enumerate(found_sources):
                if src in expected_id or expected_id.startswith(src):
                    hits += 1
                    hit = True
                    rank = idx + 1
                    mrr_sum += 1.0 / rank
                    break

        status = "✅" if hit else "❌"
        print(f"  [{i+1}/{len(dataset)}] {status} {query[:40]}... (rank={rank if hit else '-'}, {elapsed*1000:.0f}ms)")

    n = len(dataset)
    hit_rate = hits / n if n > 0 else 0
    mrr = mrr_sum / n if n > 0 else 0
    avg_latency = (total_time / n * 1000) if n > 0 else 0

    # 输出报告
    report = f"""# 知微 RAG 检索效能报告

**测试时间**: {time.strftime('%Y-%m-%d %H:%M')}
**测试集**: {dataset_path} ({n} 条)

## 核心指标

| 指标 | 得分 | 说明 |
|------|------|------|
| Hit Rate@{top_k} | {hit_rate:.1%} | 前 {top_k} 结果命中率 |
| MRR@{top_k} | {mrr:.3f} | 平均倒数排名 |
| 平均延迟 | {avg_latency:.0f}ms | 单次检索耗时 |

## 评估

- Hit Rate > 80%: {'🟢 优秀' if hit_rate > 0.8 else '🟡 需优化' if hit_rate > 0.6 else '🔴 较差'}
- MRR > 0.6: {'🟢 优秀' if mrr > 0.6 else '🟡 需优化' if mrr > 0.4 else '🔴 较差'}
- 延迟 < 500ms: {'🟢 优秀' if avg_latency < 500 else '🟡 需优化' if avg_latency < 1000 else '🔴 较差'}

## 详细结果

- 命中数: {hits}/{n}
- 总耗时: {total_time:.2f}s
"""

    # 保存报告
    reports_dir = eval_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"retrieval_{time.strftime('%Y%m%d_%H%M')}.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print("-" * 50)
    print(report)
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检索跑分")
    parser.add_argument("-d", type=str, default="golden_20.json", help="测试集文件")
    parser.add_argument("-k", type=int, default=5, help="top_k 参数")
    args = parser.parse_args()

    score_retrieval(args.d, args.k)