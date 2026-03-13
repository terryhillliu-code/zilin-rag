"""LanceDB + jieba 预分词方案验证"""
import lancedb
import jieba
import numpy as np
from pathlib import Path
import shutil

# 临时测试目录
TEST_DIR = Path("/tmp/lancedb_jieba_test")
if TEST_DIR.exists():
    shutil.rmtree(TEST_DIR)
TEST_DIR.mkdir()

print(f"LanceDB 版本: {lancedb.__version__}")
print(f"jieba 版本: {jieba.__version__}")

# 连接数据库
db = lancedb.connect(TEST_DIR)

# Mock 向量（128维）
mock_vector = np.zeros(128).tolist()

# 测试数据 - 原文 + jieba 分词
test_cases = [
    "今天苹果公司发布了最新的M4芯片，性能极强。",
    "知微系统是一个运行在macOS上的智能体架构。",
    "LangChain和LlamaIndex是两个流行的RAG框架。",
]

test_data = []
for text in test_cases:
    # jieba.cut_for_search 会产生更多的切分（搜索引擎模式）
    tokenized = " ".join(jieba.cut_for_search(text))
    print(f"\n原文: {text}")
    print(f"分词: {tokenized}")
    test_data.append({
        "text": text,                    # 原文（展示用）
        "tokenized_text": tokenized,     # 分词后（FTS 用）
        "vector": mock_vector            # 向量（语义检索用）
    })

# 创建表
table = db.create_table("test_jieba_fts", test_data)

# 在分词字段上创建 FTS 索引
try:
    table.create_fts_index("tokenized_text")
    print("\n✅ FTS 索引创建成功（在 tokenized_text 字段上）")
except Exception as e:
    print(f"\n❌ FTS 索引创建失败: {e}")
    exit(1)

# 测试搜索函数
def search_with_jieba(query: str):
    """用 jieba 分词后搜索"""
    tokenized_query = " ".join(jieba.cut(query))
    print(f"\n查询: '{query}' → 分词后: '{tokenized_query}'")
    try:
        results = table.search(tokenized_query, query_type="fts").limit(10).to_list()
        print(f"命中: {len(results)} 条")
        for r in results:
            print(f"  - {r['text'][:50]}...")
        return results
    except Exception as e:
        print(f"搜索失败: {e}")
        return []

# 测试用例
print("\n" + "="*50)
print("=== 搜索测试 ===")
print("="*50)

# 测试 1: 完整词
results_1 = search_with_jieba("苹果公司")

# 测试 2: 子词/部分匹配
results_2 = search_with_jieba("智能")

# 测试 3: 英文术语
results_3 = search_with_jieba("LangChain")

# 测试 4: 混合查询
results_4 = search_with_jieba("RAG框架")

# 结论
print("\n" + "="*50)
print("=== 验证结论 ===")
print("="*50)

success_count = 0
tests = [
    (results_1, "苹果公司", "苹果"),
    (results_2, "智能", "智能体"),
    (results_3, "LangChain", "LangChain"),
    (results_4, "RAG框架", "RAG"),
]

for results, query, expected_keyword in tests:
    if len(results) > 0:
        print(f"✅ '{query}' 命中成功")
        success_count += 1
    else:
        print(f"❌ '{query}' 命中失败")

print(f"\n总计: {success_count}/4 测试通过")

if success_count >= 3:
    print("\n🎉 方案 D 验证通过！")
    print("👉 LanceDB + jieba 预分词方案可行")
    print("👉 建议：技术栈收拢，用 LanceDB 统一向量 + FTS")
else:
    print("\n😞 方案 D 验证失败")
    print("👉 回退方案 C：SQLite FTS5 + jieba")

# 清理
shutil.rmtree(TEST_DIR)