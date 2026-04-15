#!/usr/bin/env python3
"""快速清理 LanceDB 陈旧索引"""
import os
import sys
import yaml
import lancedb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ingest.lance_store import LanceStore

def clean_stale_only():
    """只清理陈旧索引"""
    # 从配置文件读取路径
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    DB_PATH = config['paths']['lance_db']
    VAULT_ROOT = Path(config['paths']['obsidian_vault']).expanduser()

    print("🚀 开始清理陈旧索引...")

    # 直接连接 LanceDB
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table('documents')

    # 获取所有 source（使用 to_arrow）
    data = tbl.to_arrow()
    sources = set(data.column('source').to_pylist())
    print(f"📊 索引总数: {len(sources)}")

    # 找出陈旧的
    stale = []
    for s in sources:
        if s.startswith(str(VAULT_ROOT)) and not os.path.exists(s):
            stale.append(s)

    print(f"🧹 陈旧索引: {len(stale)} ({len(stale)/len(sources)*100:.1f}%)")

    # 使用 LanceStore 的 delete_by_source
    store = LanceStore(db_path=DB_PATH)

    # 执行清理
    cleaned = 0
    for s in stale:
        try:
            store.delete_by_source(s)
            cleaned += 1
            if cleaned % 100 == 0:
                print(f"  🗑️ 已清理 {cleaned}/{len(stale)}")
        except Exception as e:
            print(f"  ❌ 清理失败: {os.path.basename(s)} - {e}")

    # 验证结果
    data = tbl.to_arrow()
    remaining = len(data.column('source').to_pylist())
    print(f"\n✅ 清理完成! 清理了 {cleaned} 条，剩余 {remaining} 条有效索引")

if __name__ == "__main__":
    clean_stale_only()