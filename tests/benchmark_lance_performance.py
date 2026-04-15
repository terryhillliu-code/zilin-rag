#!/usr/bin/env python3
"""性能基准测试

测量批量索引的提速效果：
- V10.0 批量 OR SQL 删除 vs V9.0 逐个删除
- V10.0 批量 add vs V9.0 逐个 add
"""
import os
import sys
import time
import tempfile
import yaml
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
from ingest.lance_store import LanceStore, Document, escape_sql_string

# 测试配置
TEST_DB_PATH = "/tmp/benchmark_lance_db"
TEST_CHUNK_COUNT = 100  # 测试 100 个 chunk


@dataclass
class BenchmarkResult:
    name: str
    time_seconds: float
    ops_per_second: float


def setup_test_db():
    """创建测试数据库并插入测试数据"""
    if os.path.exists(TEST_DB_PATH):
        import shutil
        shutil.rmtree(TEST_DB_PATH)

    db = lancedb.connect(TEST_DB_PATH)

    # 创建测试文档
    docs = []
    for i in range(TEST_CHUNK_COUNT):
        doc = Document(
            id=f"test_{i}",
            text=f"Test content {i}",
            raw_text=f"Test content {i}",
            source=f"/test/file_{i}.md",
            filename=f"file_{i}.md",
            h1="", h2="",
            category="",
            tags="",
            char_count=10,
            vector=[0.0] * 1024,  # 假设 1024 维向量
            tokenized_text="test content"
        )
        docs.append(doc)

    # 创建表并插入
    tbl = db.create_table("documents", data=[doc.__dict__ for doc in docs])
    return db, tbl


def benchmark_batch_delete(tbl):
    """基准：批量 OR SQL 删除"""
    sources = [f"/test/file_{i}.md" for i in range(TEST_CHUNK_COUNT)]

    start = time.time()
    sql = " OR ".join([f"source = '{escape_sql_string(s)}'" for s in sources])
    tbl.delete(sql)
    elapsed = time.time() - start

    return BenchmarkResult(
        name="批量 OR SQL 删除",
        time_seconds=elapsed,
        ops_per_second=TEST_CHUNK_COUNT / elapsed
    )


def benchmark_individual_delete(tbl):
    """基准：逐个删除（模拟 V9.0）"""
    # 重新插入数据
    docs = []
    for i in range(TEST_CHUNK_COUNT):
        doc = Document(
            id=f"test_{i}",
            text=f"Test content {i}",
            raw_text=f"Test content {i}",
            source=f"/test/file_{i}.md",
            filename=f"file_{i}.md",
            h1="", h2="",
            category="",
            tags="",
            char_count=10,
            vector=[0.0] * 1024,
            tokenized_text="test content"
        )
        docs.append(doc.__dict__)
    tbl.add(docs)

    sources = [f"/test/file_{i}.md" for i in range(TEST_CHUNK_COUNT)]

    start = time.time()
    for s in sources:
        tbl.delete(f"source = '{escape_sql_string(s)}'")
    elapsed = time.time() - start

    return BenchmarkResult(
        name="逐个删除 (V9.0)",
        time_seconds=elapsed,
        ops_per_second=TEST_CHUNK_COUNT / elapsed
    )


def benchmark_batch_add(tbl):
    """基准：批量 add"""
    docs = []
    for i in range(TEST_CHUNK_COUNT):
        doc = Document(
            id=f"test_batch_{i}",
            text=f"Test content {i}",
            raw_text=f"Test content {i}",
            source=f"/test/batch_{i}.md",
            filename=f"batch_{i}.md",
            h1="", h2="",
            category="",
            tags="",
            char_count=10,
            vector=[0.0] * 1024,
            tokenized_text="test content"
        )
        docs.append(doc.__dict__)

    start = time.time()
    tbl.add(docs)
    elapsed = time.time() - start

    return BenchmarkResult(
        name="批量 add",
        time_seconds=elapsed,
        ops_per_second=TEST_CHUNK_COUNT / elapsed
    )


def benchmark_individual_add(tbl):
    """基准：逐个 add（模拟 V9.0）"""
    # 清空之前的 batch 数据
    tbl.delete("source LIKE '/test/batch_%'")

    start = time.time()
    for i in range(TEST_CHUNK_COUNT):
        doc = Document(
            id=f"test_ind_{i}",
            text=f"Test content {i}",
            raw_text=f"Test content {i}",
            source=f"/test/ind_{i}.md",
            filename=f"ind_{i}.md",
            h1="", h2="",
            category="",
            tags="",
            char_count=10,
            vector=[0.0] * 1024,
            tokenized_text="test content"
        )
        tbl.add([doc.__dict__])
    elapsed = time.time() - start

    return BenchmarkResult(
        name="逐个 add (V9.0)",
        time_seconds=elapsed,
        ops_per_second=TEST_CHUNK_COUNT / elapsed
    )


def run_benchmarks():
    """运行所有基准测试"""
    print("=" * 60)
    print("LanceDB 性能基准测试")
    print(f"测试规模: {TEST_CHUNK_COUNT} 个 chunk")
    print("=" * 60)
    print()

    db, tbl = setup_test_db()
    results = []

    # 删除测试
    print("[删除测试]")
    results.append(benchmark_batch_delete(tbl))
    results.append(benchmark_individual_delete(tbl))

    # 重新填充数据
    docs = []
    for i in range(TEST_CHUNK_COUNT):
        doc = Document(
            id=f"test_{i}",
            text=f"Test content {i}",
            raw_text=f"Test content {i}",
            source=f"/test/file_{i}.md",
            filename=f"file_{i}.md",
            h1="", h2="",
            category="",
            tags="",
            char_count=10,
            vector=[0.0] * 1024,
            tokenized_text="test content"
        )
        docs.append(doc.__dict__)
    tbl.add(docs)

    # 添加测试
    print("[添加测试]")
    results.append(benchmark_batch_add(tbl))
    results.append(benchmark_individual_add(tbl))

    # 输出结果
    print()
    print("=" * 60)
    print("基准测试结果")
    print("=" * 60)
    print()

    # 按类型分组比较
    delete_results = [r for r in results if "删除" in r.name]
    add_results = [r for r in results if "add" in r.name or "添加" in r.name]

    print("【删除操作对比】")
    for r in delete_results:
        print(f"  {r.name}: {r.time_seconds:.3f}s ({r.ops_per_second:.1f} ops/s)")

    batch_del = next(r for r in delete_results if "批量" in r.name)
    ind_del = next(r for r in delete_results if "逐个" in r.name)
    speedup_del = ind_del.time_seconds / batch_del.time_seconds
    print(f"  → 批量删除提速: {speedup_del:.1f}x")

    print()
    print("【添加操作对比】")
    for r in add_results:
        print(f"  {r.name}: {r.time_seconds:.3f}s ({r.ops_per_second:.1f} ops/s)")

    batch_add = next(r for r in add_results if "批量" in r.name)
    ind_add = next(r for r in add_results if "逐个" in r.name)
    speedup_add = ind_add.time_seconds / batch_add.time_seconds
    print(f"  → 批量添加提速: {speedup_add:.1f}x")

    print()
    print("=" * 60)
    print(f"总体预期提速: {speedup_del * speedup_add:.1f}x")
    print("=" * 60)

    # 清理
    import shutil
    shutil.rmtree(TEST_DB_PATH, ignore_errors=True)


if __name__ == "__main__":
    run_benchmarks()