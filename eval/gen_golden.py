# ~/zhiwei-rag/eval/gen_golden.py
"""
黄金测试集生成器
- 从 LanceDB 随机抽取 chunk
- 调 LLM 反向生成问题
- 输出 JSON 测试集
"""
import json
import random
import sys
import argparse
from pathlib import Path

# 添加 zhiwei-bot 到路径
sys.path.insert(0, str(Path.home() / "zhiwei-bot"))

from core.llm_client import llm_client
import lancedb


def generate_golden_dataset(n: int = 20, output: str = "golden_20.json"):
    """
    生成黄金测试集

    Args:
        n: 抽取样本数量
        output: 输出文件名
    """
    # 1. 连接 LanceDB
    rag_dir = Path.home() / "zhiwei-rag"
    db = lancedb.connect(str(rag_dir / "data/lance_db"))
    table = db.open_table("documents")

    # 2. 随机抽取 n 个 chunk
    print(f"从 {table.count_rows()} 条记录中抽取 {n} 条...")
    all_docs = table.to_arrow()

    # 随机选择索引
    total = all_docs.num_rows
    sample_indices = random.sample(range(total), min(n, total))

    # 3. 调 LLM 生成问题
    dataset = []

    for i, idx in enumerate(sample_indices):
        # 提取字段
        text = all_docs.column("text")[idx].as_py()
        doc_id = all_docs.column("id")[idx].as_py()
        source = all_docs.column("source")[idx].as_py()

        content = text[:500] if text else ""  # 取前 500 字

        prompt = f"""根据以下文本内容，生成一个具体的、有明确答案的问题。

要求：
1. 问题必须能用这段文本直接回答
2. 问题要包含具体的名词或数字，不要问"是什么分类"这种泛化问题
3. 优先问具体事实：数量、方法、性能、时间、技术术语等
4. 禁止问元数据问题：不要问文件大小、文档编号、文件名、创建时间等
5. 问题应该关注文本内容本身的知识点
6. 只输出问题，不要其他内容

文本内容：
{content}
"""
        success, question = llm_client.call("chat", prompt)

        if not success:
            print(f"  [{i+1}/{n}] LLM 调用失败，跳过")
            continue

        dataset.append({
            "query": question.strip(),
            "expected_id": doc_id,
            "expected_source": source,
            "expected_content": content[:200],
        })
        print(f"  [{i+1}/{n}] 已生成: {question.strip()[:50]}...")

    # 4. 保存
    output_path = Path(__file__).parent / output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n生成 {len(dataset)} 个测试用例 → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成黄金测试集")
    parser.add_argument("-n", type=int, default=20, help="样本数量")
    parser.add_argument("-o", type=str, default="golden_20.json", help="输出文件名")
    args = parser.parse_args()

    generate_golden_dataset(args.n, args.o)