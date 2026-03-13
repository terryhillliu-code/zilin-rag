"""LanceDB 中文 FTS 能力验证"""
import lancedb
import numpy as np
from pathlib import Path
import shutil

# 临时测试目录
TEST_DIR = Path("/tmp/lancedb_fts_test")
if TEST_DIR.exists():
    shutil.rmtree(TEST_DIR)
TEST_DIR.mkdir()

print(f"LanceDB 版本: {lancedb.__version__}")

# 连接数据库
db = lancedb.connect(TEST_DIR)

# Mock 向量（128维，重点测 FTS）
mock_vector = np.zeros(128).tolist()

# 测试数据
test_data = [
    {"text": "今天苹果公司发布了最新的M4芯片，性能极强。", "vector": mock_vector},
    {"text": "知微系统是一个运行在 macOS 上的智能体架构。", "vector": mock_vector},
]

# 创建表
table = db.create_table("test_fts", test_data)

# 创建 FTS 索引
try:
    # 尝试创建全文索引（检查是否支持中文分词器）
    table.create_fts_index("text")
    print("✅ FTS 索引创建成功")
except Exception as e:
    print(f"❌ FTS 索引创建失败: {e}")
    exit(1)

# 测试搜索
print("\n=== 搜索测试 ===")

# 测试 A: 完整词
query_a = "苹果公司"
try:
    results_a = table.search(query_a, query_type="fts").limit(10).to_list()
    print(f"\n搜索 '{query_a}': 命中 {len(results_a)} 条")
    for r in results_a:
        print(f"  - {r['text'][:50]}...")
except Exception as e:
    print(f"搜索 '{query_a}' 失败: {e}")
    results_a = []

# 测试 B: 子词/部分匹配
query_b = "智能"
try:
    results_b = table.search(query_b, query_type="fts").limit(10).to_list()
    print(f"\n搜索 '{query_b}': 命中 {len(results_b)} 条")
    for r in results_b:
        print(f"  - {r['text'][:50]}...")
except Exception as e:
    print(f"搜索 '{query_b}' 失败: {e}")
    results_b = []

# 结论
print("\n=== 验证结论 ===")
if len(results_b) > 0 and "智能体" in results_b[0]["text"]:
    print("✅ LanceDB 中文 FTS 验证通过！子词 '智能' 成功命中 '智能体架构'")
    print("👉 方案 B 绿灯通行")
else:
    print("❌ LanceDB 中文 FTS 验证失败！子词无法命中")
    print("👉 方案 B 不可行，需使用 jieba 预处理方案")

# 清理
shutil.rmtree(TEST_DIR)