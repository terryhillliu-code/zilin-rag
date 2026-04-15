"""LanceStore 工具函数测试"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.lance_store import escape_sql_string


def test_escape_sql_string_basic():
    """测试基本单引号转义"""
    assert escape_sql_string("test'file.md") == "test''file.md"


def test_escape_sql_string_multiple():
    """测试多个单引号转义"""
    assert escape_sql_string("a'b'c.md") == "a''b''c.md"


def test_escape_sql_string_no_quote():
    """测试无单引号字符串"""
    assert escape_sql_string("normal.md") == "normal.md"


def test_escape_sql_string_empty():
    """测试空字符串"""
    assert escape_sql_string("") == ""


def test_escape_sql_string_only_quote():
    """测试仅包含单引号"""
    assert escape_sql_string("'") == "''"


def test_batch_delete_sql_construction():
    """测试批量删除 SQL 构建"""
    sources = ["file1.md", "file'2.md", "file3.md"]
    conditions = [f"source = '{escape_sql_string(s)}'" for s in sources]
    sql = " OR ".join(conditions)

    # 验证 SQL 格式正确
    assert "source = 'file1.md'" in sql
    assert "source = 'file''2.md'" in sql
    assert "source = 'file3.md'" in sql
    assert sql.count(" OR ") == 2  # 三个条件，两个 OR


if __name__ == "__main__":
    test_escape_sql_string_basic()
    test_escape_sql_string_multiple()
    test_escape_sql_string_no_quote()
    test_escape_sql_string_empty()
    test_escape_sql_string_only_quote()
    test_batch_delete_sql_construction()
    print("✓ 所有测试通过")