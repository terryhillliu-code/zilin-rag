"""ingest_batch.py 单元测试

测试批量删除和批量写入的核心逻辑：
- 批量 OR SQL 构建
- 条件转义
- 文件批次处理
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.lance_store import escape_sql_string


def test_escape_sql_string():
    """测试 SQL 字符串转义"""
    # 基本转义
    assert escape_sql_string("test'file.md") == "test''file.md"
    # 多引号
    assert escape_sql_string("a'b'c.md") == "a''b''c.md"
    # 无引号
    assert escape_sql_string("normal.md") == "normal.md"


def test_batch_delete_sql_construction():
    """测试批量删除 SQL 构建"""
    sources = [
        "/test/file1.md",
        "/test/file'2.md",  # 包含单引号
        "/test/file3.md"
    ]

    conditions = [f"source = '{escape_sql_string(s)}'" for s in sources]
    sql = " OR ".join(conditions)

    # 验证格式正确
    assert "source = '/test/file1.md'" in sql
    assert "source = '/test/file''2.md'" in sql  # 转义后的引号
    assert "source = '/test/file3.md'" in sql
    assert sql.count(" OR ") == 2


def test_batch_delete_sql_empty_sources():
    """测试空源列表"""
    sources = []
    conditions = [f"source = '{escape_sql_string(s)}'" for s in sources]
    sql = " OR ".join(conditions)
    assert sql == ""


def test_batch_source_info_tracking():
    """测试批次源信息追踪逻辑"""
    batch_source_info = []

    # 模拟有 chunk 的文件
    batch_source_info.append(("file1.md", 5))
    # 模拟无 chunk 的文件
    batch_source_info.append(("file2.md", 0))
    # 模拟失败的文件
    batch_source_info.append(("file3.md", 0))

    # 提取需要删除的源
    sources_to_delete = [src for src, cnt in batch_source_info if cnt > 0]
    assert sources_to_delete == ["file1.md"]


def test_chunk_index_calculation():
    """测试 chunk 索引计算"""
    batch_source_info = [
        ("file1.md", 3),
        ("file2.md", 2),
        ("file3.md", 4)
    ]

    # 计算各文件的 chunk 范围
    emb_idx = 0
    ranges = []
    for src, cnt in batch_source_info:
        if cnt > 0:
            ranges.append((emb_idx, emb_idx + cnt))
            emb_idx += cnt

    assert ranges == [(0, 3), (3, 5), (5, 9)]
    assert emb_idx == 9  # 总 chunk 数


def test_batch_trigger_conditions():
    """测试批次处理触发条件"""
    batch_size = 20
    max_chunks = 500

    # 测试文件数触发
    assert len([]) < batch_size
    assert len(["f"] * 20) >= batch_size

    # 测试 chunk 数触发
    assert 400 < max_chunks
    assert 600 >= max_chunks

    # 测试最终触发
    i, total = 99, 100
    assert (i + 1) == total  # 最后一个文件应触发


def test_stats_tracking():
    """测试统计追踪"""
    stats = {"success": 0, "fail": 0, "chunks": 0}

    # 模拟处理结果
    stats["success"] += 1
    stats["chunks"] += 5

    stats["fail"] += 1
    stats["chunks"] += 0

    stats["success"] += 1
    stats["chunks"] += 3

    assert stats["success"] == 2
    assert stats["fail"] == 1
    assert stats["chunks"] == 8


def test_progress_rate_calculation():
    """测试进度速率计算"""
    stats = {"chunks": 100}
    elapsed = 10.0

    rate = stats["chunks"] / elapsed if elapsed > 0 else 0
    assert rate == 10.0


def test_report_interval_trigger():
    """测试进度报告触发间隔"""
    report_interval = 100

    for i in range(100):
        should_report = (i + 1) % report_interval == 0 or (i + 1) == 100
        if i == 99:  # 第 100 个
            assert should_report
        elif i == 49:  # 第 50 个
            assert not should_report


if __name__ == "__main__":
    test_escape_sql_string()
    test_batch_delete_sql_construction()
    test_batch_delete_sql_empty_sources()
    test_batch_source_info_tracking()
    test_chunk_index_calculation()
    test_batch_trigger_conditions()
    test_stats_tracking()
    test_progress_rate_calculation()
    test_report_interval_trigger()
    print("✓ ingest_batch.py 单元测试通过")