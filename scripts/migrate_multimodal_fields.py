#!/usr/bin/env python3
"""
MM-001 迁移脚本：为 LanceDB documents 添加多模态字段

新增字段：
- chunk_type: string (默认 "text")
- page: int (默认 0)
- timestamp: string (默认 "")
- figure_path: string (默认 "")

流程：
1. 读取现有 documents 表所有数据
2. 为每条记录添加默认值
3. 重建表（添加新字段）
4. 写回数据
5. 重建 FTS 索引

使用：
    cd ~/zhiwei-rag && source venv/bin/activate
    python scripts/migrate_multimodal_fields.py
"""
import sys
import time
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
import pyarrow as pa
import fcntl
from contextlib import contextmanager


@contextmanager
def write_lock(lock_path: str):
    """文件锁上下文管理器"""
    lock_f = open(lock_path, "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_f, fcntl.LOCK_UN)
        lock_f.close()


def migrate():
    """执行迁移"""
    db_path = Path(__file__).parent.parent / "data" / "lance_db"
    db_path = db_path.expanduser()
    lock_file = db_path / "write.lock"

    print(f"[迁移] LanceDB 路径: {db_path}")

    # 连接数据库
    db = lancedb.connect(str(db_path))

    if "documents" not in db.table_names():
        print("[错误] documents 表不存在")
        return False

    table = db.open_table("documents")
    total_count = table.count_rows()
    print(f"[迁移] 现有文档数: {total_count}")

    # 检查是否已有新字段
    schema = table.schema
    field_names = [f.name for f in schema]

    new_fields = ["chunk_type", "page", "timestamp", "figure_path"]
    existing_new_fields = [f for f in new_fields if f in field_names]

    if existing_new_fields:
        print(f"[迁移] 已存在字段: {existing_new_fields}")

        if len(existing_new_fields) == len(new_fields):
            print("[迁移] 所有新字段已存在，检查数据...")

            # 检查 chunk_type 默认值
            df = table.to_lance().to_table(columns=['id', 'chunk_type']).to_pandas()
            text_count = (df['chunk_type'] == 'text').sum()
            print(f"[迁移] chunk_type='text': {text_count}/{total_count}")

            print("[迁移] 跳过迁移")
            return True

    print("[迁移] 开始迁移...")

    # 使用文件锁保护
    with write_lock(str(lock_file)):
        # 1. 读取所有数据
        print("[迁移] 步骤 1/5: 读取现有数据...")
        all_data = table.to_lance().to_table().to_pydict()

        # 2. 添加新字段默认值
        print("[迁移] 步骤 2/5: 添加多模态字段...")
        all_data['chunk_type'] = ['text'] * total_count
        all_data['page'] = [0] * total_count
        all_data['timestamp'] = [''] * total_count
        all_data['figure_path'] = [''] * total_count

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
            # 多模态字段
            pa.field("chunk_type", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("timestamp", pa.string()),
            pa.field("figure_path", pa.string()),
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
                'tokenized_text': all_data['tokenized_text'][i] or '',
                'vector': all_data['vector'][i],
                # 多模态字段
                'chunk_type': all_data['chunk_type'][i],
                'page': all_data['page'][i],
                'timestamp': all_data['timestamp'][i],
                'figure_path': all_data['figure_path'][i],
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
    field_names = [f.name for f in schema]
    print(f"[验证] Schema 字段: {field_names}")

    # 检查记录数
    count = table.count_rows()
    print(f"[验证] 记录数: {count}")

    # 检查新字段
    new_fields = ["chunk_type", "page", "timestamp", "figure_path"]
    for field in new_fields:
        if field in field_names:
            print(f"[验证] ✓ {field} 字段存在")
        else:
            print(f"[验证] ✗ {field} 字段缺失")

    # 检查 chunk_type 默认值
    df = table.to_lance().to_table(columns=['id', 'chunk_type', 'page']).to_pandas()
    text_count = (df['chunk_type'] == 'text').sum()
    print(f"[验证] chunk_type='text': {text_count}/{count}")

    # 测试 FTS 搜索
    print("\n[验证] 测试 FTS 搜索...")
    try:
        results = table.search("智能", query_type="fts").limit(3).to_list()
        print(f"[验证] FTS 搜索命中: {len(results)} 条")
        if results:
            keys = list(results[0].keys())
            print(f"[验证] 结果字段: {keys}")
    except Exception as e:
        print(f"[验证] FTS 搜索失败: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MM-001 多模态字段迁移脚本")
    parser.add_argument("--verify-only", action="store_true", help="仅验证，不执行迁移")
    args = parser.parse_args()

    if args.verify_only:
        verify()
    else:
        success = migrate()
        if success:
            verify()