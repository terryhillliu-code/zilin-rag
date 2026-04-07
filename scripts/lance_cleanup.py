#!/usr/bin/env python3
"""
LanceDB 清理脚本

定期执行 optimize 清理事务文件和旧版本
v72.3: 更激进的清理策略（1天），支持多表
"""
import os
import sys
from pathlib import Path
from datetime import timedelta
import shutil

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb

DB_PATH = Path.home() / "zhiwei-rag" / "data" / "lance_db"


def get_db_size():
    """获取数据库总大小"""
    total = 0
    for item in DB_PATH.iterdir():
        if item.is_dir():
            for sub in item.rglob("*"):
                if sub.is_file():
                    total += sub.stat().st_size
    return total / (1024 * 1024 * 1024)  # GB


def get_stats():
    """获取数据库统计"""
    stats = {}
    for table_dir in DB_PATH.glob("*.lance"):
        table_name = table_dir.stem
        transactions_dir = table_dir / "_transactions"
        versions_dir = table_dir / "_versions"

        stats[table_name] = {
            "transactions": len(list(transactions_dir.glob("*"))) if transactions_dir.exists() else 0,
            "versions": len(list(versions_dir.glob("*"))) if versions_dir.exists() else 0,
        }
    return stats


def cleanup(days: int = 1, aggressive: bool = True):
    """执行清理

    Args:
        days: 保留最近 N 天的版本
        aggressive: 是否执行激进清理（compact + cleanup）
    """
    print(f"[LanceDB] 开始清理，保留最近 {days} 天版本...")

    try:
        db = lancedb.connect(str(DB_PATH))

        # 使用新版 API
        table_names = list(db.list_tables())
        print(f"[LanceDB] 发现 {len(table_names)} 个表: {table_names}")

        before_size = get_db_size()
        before_stats = get_stats()
        print(f"[LanceDB] 清理前大小: {before_size:.2f} GB")

        for table_name in table_names:
            print(f"\n[LanceDB] 处理表: {table_name}")
            table = db.open_table(table_name)

            print(f"  - 文档数: {len(table)}")

            # 执行 optimize
            print(f"  - 执行 optimize...")
            table.optimize(cleanup_older_than=timedelta(days=days))

            if aggressive:
                # 激进清理：强制 compact
                try:
                    table.compact()
                    print(f"  - 执行 compact 完成")
                except Exception as e:
                    print(f"  - compact 失败 (可能已是最优): {e}")

        after_size = get_db_size()
        after_stats = get_stats()

        print(f"\n[LanceDB] 清理后大小: {after_size:.2f} GB")
        print(f"[LanceDB] 释放空间: {before_size - after_size:.2f} GB")

        # 统计清理效果
        total_cleaned = 0
        for table_name in table_names:
            if table_name in before_stats and table_name in after_stats:
                cleaned = before_stats[table_name]["transactions"] - after_stats[table_name]["transactions"]
                total_cleaned += cleaned
                print(f"  - {table_name}: 删除 {cleaned} 个事务文件")

        print(f"[LanceDB] ✅ 清理完成，共删除 {total_cleaned} 个事务文件")

        return True

    except Exception as e:
        print(f"[LanceDB] ❌ 清理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LanceDB 清理")
    parser.add_argument("--days", type=int, default=1, help="保留最近 N 天的版本")
    parser.add_argument("--stats", action="store_true", help="仅显示统计")
    parser.add_argument("--aggressive", action="store_true", default=True, help="激进清理模式")

    args = parser.parse_args()

    if args.stats:
        size = get_db_size()
        stats = get_stats()
        print(f"数据库大小: {size:.2f} GB")
        for table_name, table_stats in stats.items():
            print(f"{table_name}: {table_stats['transactions']} 事务, {table_stats['versions']} 版本")
    else:
        cleanup(args.days, args.aggressive)
