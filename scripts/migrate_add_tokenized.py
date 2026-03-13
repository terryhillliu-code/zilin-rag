#!/usr/bin/env python3
"""
FTS-001 迁移脚本：为 LanceDB documents 添加 tokenized_text 字段

功能：
1. 读取现有 documents 表所有数据
2. 为每条记录计算 tokenized_text（jieba 分词）
3. 重建表（添加新字段）
4. 写回数据
5. 创建 FTS 索引

使用：
    cd ~/zhiwei-rag && source venv/bin/activate
    python scripts/migrate_add_tokenized.py
"""
import sys
import time
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
import pyarrow as pa
import jieba
from tqdm import tqdm


def tokenize_text(text: str) -> str:
    """jieba 分词（搜索引擎模式）"""
    if not text:
        return ""
    return " ".join(jieba.cut_for_search(text))


def migrate():
    """执行迁移"""
    db_path = Path(__file__).parent.parent / "data" / "lance_db"
    db_path = db_path.expanduser()

    print(f"[迁移] LanceDB 路径: {db_path}")

    # 连接数据库
    db = lancedb.connect(str(db_path))

    if "documents" not in db.table_names():
        print("[错误] documents 表不存在")
        return False

    table = db.open_table("documents")
    total_count = table.count_rows()
    print(f"[迁移] 现有文档数: {total_count}")

    # 检查是否已有 tokenized_text 字段
    schema = table.schema
    field_names = [f.name for f in schema]

    if "tokenized_text" in field_names:
        print("[迁移] tokenized_text 字段已存在，检查数据...")

        # 检查是否所有记录都有值
        df = table.to_lance().to_table(columns=['id', 'tokenized_text']).to_pandas()
        non_empty = (df['tokenized_text'] != '').sum()
        print(f"[迁移] tokenized_text 非空数: {non_empty}/{total_count}")

        if non_empty == total_count:
            print("[迁移] 所有记录已有 tokenized_text，跳过迁移")
            # 尝试创建 FTS 索引
            try:
                table.create_fts_index("tokenized_text")
                print("[迁移] FTS 索引创建成功")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print("[迁移] FTS 索引已存在")
                else:
                    print(f"[迁移] FTS 索引创建失败: {e}")
            return True

    print("[迁移] 开始迁移...")

    # 1. 读取所有数据
    print("[迁移] 步骤 1/5: 读取现有数据...")
    all_data = table.to_lance().to_table().to_pydict()

    # 2. 计算 tokenized_text
    print("[迁移] 步骤 2/5: 计算 tokenized_text...")
    raw_texts = all_data.get('raw_text', all_data.get('text', [''] * total_count))
    tokenized_texts = []

    for raw_text in tqdm(raw_texts, desc="分词中"):
        tokenized_texts.append(tokenize_text(raw_text or ""))

    all_data['tokenized_text'] = tokenized_texts

    # 3. 准备新 schema
    print("[迁移] 步骤 3/5: 准备新 schema...")

    # 获取向量维度
    vectors = all_data.get('vector', [[]])
    dim = 1024
    if vectors and len(vectors) > 0 and len(vectors[0]) > 0:
        dim = len(vectors[0])

    new_schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("raw_text", pa.string()),
        pa.field("source", pa.string()),
        pa.field("filename", pa.string()),
        pa.field("h1", pa.string()),
        pa.field("h2", pa.string()),
        pa.field("category", pa.string()),
        pa.field("tags", pa.string()),
        pa.field("char_count", pa.int32()),
        pa.field("tokenized_text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])

    # 4. 删除旧表，创建新表
    print("[迁移] 步骤 4/5: 重建表...")

    # 转换为正确格式
    records = []
    for i in range(total_count):
        record = {
            'id': all_data['id'][i],
            'text': all_data['text'][i],
            'raw_text': all_data['raw_text'][i],
            'source': all_data['source'][i],
            'filename': all_data['filename'][i],
            'h1': all_data['h1'][i] or '',
            'h2': all_data['h2'][i] or '',
            'category': all_data['category'][i] or '',
            'tags': all_data['tags'][i] or '',
            'char_count': all_data['char_count'][i] or 0,
            'tokenized_text': all_data['tokenized_text'][i],
            'vector': all_data['vector'][i],
        }
        records.append(record)

    # 删除旧表
    db.drop_table("documents")

    # 创建新表
    new_table = db.create_table(
        "documents",
        data=records,
        schema=new_schema,
        mode="create"
    )

    print(f"[迁移] 新表创建完成，文档数: {new_table.count_rows()}")

    # 5. 创建 FTS 索引
    print("[迁移] 步骤 5/5: 创建 FTS 索引...")

    try:
        new_table.create_fts_index("tokenized_text")
        print("[迁移] FTS 索引创建成功")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("[迁移] FTS 索引已存在")
        else:
            print(f"[迁移] FTS 索引创建失败: {e}")

    print("[迁移] 迁移完成!")
    return True


def verify():
    """验证迁移结果"""
    db_path = Path(__file__).parent.parent / "data" / "lance_db"
    db_path = db_path.expanduser()

    db = lancedb.connect(str(db_path))
    table = db.open_table("documents")

    print("\n[验证] 检查迁移结果...")

    # 检查 schema
    schema = table.schema
    print(f"[验证] Schema 字段: {[f.name for f in schema]}")

    # 检查 tokenized_text
    df = table.to_lance().to_table(columns=['id', 'tokenized_text']).to_pandas()
    non_empty = (df['tokenized_text'] != '').sum()
    print(f"[验证] tokenized_text 非空: {non_empty}/{len(df)}")

    # 测试 FTS 搜索
    print("\n[验证] 测试 FTS 搜索...")

    test_queries = ["智能", "LangChain", "向量"]

    for query in test_queries:
        tokenized_query = tokenize_text(query)
        print(f"  查询: '{query}' → 分词: '{tokenized_query}'")

        try:
            results = table.search(tokenized_query, query_type="fts").limit(3).to_list()
            print(f"    命中: {len(results)} 条")
            if results:
                print(f"    Top1: {results[0]['id'][:50]}...")
        except Exception as e:
            print(f"    错误: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FTS-001 迁移脚本")
    parser.add_argument("--verify-only", action="store_true", help="仅验证，不执行迁移")
    args = parser.parse_args()

    if args.verify_only:
        verify()
    else:
        success = migrate()
        if success:
            verify()