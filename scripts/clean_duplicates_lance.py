#!/usr/bin/env python3
"""清理LanceDB重复索引

策略：
1. 找出所有有重复ID的source
2. 批量删除这些source的所有记录
3. 记录删除的source，供后续补全使用

注意：删除后这些文件的索引会暂时缺失，需要在补全步骤中重新索引
"""
import sys
import time
import yaml
from pathlib import Path
from collections import Counter
import lancedb

sys.path.insert(0, str(Path(__file__).parent.parent))
from ingest.lance_store import escape_sql_string

def clean_duplicates():
    """清理重复索引"""
    # 从配置文件读取路径
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    DB_PATH = config['paths']['lance_db']
    VAULT_ROOT = str(Path(config['paths']['obsidian_vault']).expanduser())

    print("🧹 开始清理重复索引...")

    db = lancedb.connect(DB_PATH)
    tbl = db.open_table('documents')
    data = tbl.to_arrow()

    all_ids = list(data.column('id').to_pylist())
    all_sources = list(data.column('source').to_pylist())

    # 统计重复ID
    id_counts = Counter(all_ids)
    dup_ids = set(id for id, c in id_counts.items() if c > 1)

    print(f"总记录数: {len(all_ids)}")
    print(f"重复ID数: {len(dup_ids)}")
    print(f"重复记录数: {sum(c-1 for c in id_counts.values() if c > 1)}")

    # 找出有重复的source
    dup_sources = set()
    for i, id_val in enumerate(all_ids):
        if id_val in dup_ids:
            dup_sources.add(all_sources[i])

    print(f"涉及重复的source数: {len(dup_sources)}")

    # 过滤出Vault的source（非Library）
    vault_dup_sources = [s for s in dup_sources if s.startswith(VAULT_ROOT)]
    library_dup_sources = [s for s in dup_sources if not s.startswith(VAULT_ROOT)]

    print(f"Vault重复source: {len(vault_dup_sources)}")
    print(f"Library重复source: {len(library_dup_sources)}")

    # 执行批量删除（使用OR条件）
    print("\n开始删除重复source的索引...")
    start_time = time.time()

    # 分批删除（每批100个）
    batch_size = 100
    deleted_count = 0

    all_dup_sources = list(dup_sources)
    for i in range(0, len(all_dup_sources), batch_size):
        batch = all_dup_sources[i:i+batch_size]
        conditions = []
        for s in batch:
            conditions.append(f"source = '{escape_sql_string(s)}'")

        sql = " OR ".join(conditions)
        try:
            tbl.delete(sql)
            deleted_count += len(batch)
            if deleted_count % 500 == 0 or deleted_count == len(all_dup_sources):
                print(f"  已删除 {deleted_count}/{len(all_dup_sources)} source", flush=True)
        except Exception as e:
            print(f"  批次删除失败: {e}")
            # 降级为逐个删除
            for s in batch:
                try:
                    tbl.delete(f"source = '{escape_sql_string(s)}'")
                except:
                    pass

    elapsed = time.time() - start_time
    print(f"删除耗时: {elapsed:.1f}s")

    # 验证结果
    data = tbl.to_arrow()
    remaining = len(data.column('id').to_pylist())
    print(f"\n剩余记录数: {remaining}")

    # 保存需要重新索引的source列表
    if vault_dup_sources:
        reindex_path = '/tmp/reindex_after_duplicate_clean.txt'
        with open(reindex_path, 'w') as f:
            for s in vault_dup_sources:
                if Path(s).exists():  # 只保存存在的文件
                    f.write(s + '\n')
        print(f"需要重新索引的Vault文件: {len([s for s in vault_dup_sources if Path(s).exists()])}")
        print(f"已保存到: {reindex_path}")

    return {
        'deleted_sources': len(all_dup_sources),
        'remaining_records': remaining,
        'elapsed': elapsed
    }


if __name__ == "__main__":
    clean_duplicates()