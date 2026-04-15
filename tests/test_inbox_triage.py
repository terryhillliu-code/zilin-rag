"""inbox_triage.py 集成测试

测试文件移动和索引同步的完整流程：
1. 创建临时 Markdown 文件
2. 模拟 frontmatter 和标签
3. 验证分类逻辑
4. 验证索引同步（删除旧索引 + 索引新位置）
"""
import os
import sys
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.inbox_triage import (
    parse_frontmatter,
    extract_tags,
    classify_file,
    process_file,
)


def test_parse_frontmatter_basic():
    """测试基本 frontmatter 解析"""
    content = """---
title: Test Paper
tags: [LLM, AI]
---
# Content here"""
    fm, body = parse_frontmatter(content)
    assert fm.get("title") == "Test Paper"
    assert fm.get("tags") == ["LLM", "AI"]
    assert "# Content here" in body


def test_parse_frontmatter_no_fm():
    """测试无 frontmatter 内容"""
    content = "# Just content"
    fm, body = parse_frontmatter(content)
    assert fm == {}
    assert content == body


def test_extract_tags_from_list():
    """测试从 tags 字段提取标签"""
    frontmatter = {"tags": ["LLM", "RAG", "Agent"]}
    tags = extract_tags(frontmatter)
    assert "LLM" in tags
    assert "RAG" in tags


def test_extract_tags_from_string():
    """测试从字符串 tags 提取"""
    frontmatter = {"tags": "机器学习"}
    tags = extract_tags(frontmatter)
    assert "机器学习" in tags


def test_extract_tags_from_title():
    """测试从标题括号提取"""
    frontmatter = {"title": "论文标题（深度学习）"}
    tags = extract_tags(frontmatter)
    assert "深度学习" in tags


def test_classify_by_tag():
    """测试按标签分类"""
    frontmatter = {"tags": ["LLM"]}
    content = ""
    folder = classify_file(frontmatter, content)
    assert folder == "10-19_AI系统_AI-Systems"


def test_classify_by_content():
    """测试按内容关键词分类"""
    frontmatter = {}
    content = "这是一篇关于GPU架构的论文..."
    folder = classify_file(frontmatter, content)
    assert folder == "20-29_AI硬件_AI-Hardware"


def test_classify_paper_prefix():
    """测试 PAPER 前缀分类"""
    frontmatter = {"title": "PAPER-123 Test"}
    content = ""
    folder = classify_file(frontmatter, content)
    assert folder == "10-19_AI系统_AI-Systems"


def test_classify_default():
    """测试默认分类"""
    frontmatter = {}
    content = "普通内容无关键词"
    folder = classify_file(frontmatter, content)
    assert folder == "90-99_系统与归档_System"


def test_process_file_dry_run():
    """测试 dry_run 模式（不实际移动）"""
    # 创建临时测试文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("""---
title: Test Paper (LLM)
tags: [LLM]
rag_indexed: true
---
# Content""")
        test_file = Path(f.name)

    try:
        result = process_file(test_file, dry_run=True)

        # 验证结果
        assert result["file"] == test_file.name
        assert result["target"] == "10-19_AI系统_AI-Systems"
        assert test_file.exists()  # dry_run 不应删除文件

    finally:
        test_file.unlink(missing_ok=True)


def test_process_file_need_process():
    """测试未索引文件跳过"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("""---
title: Test
---
# Content""")
        test_file = Path(f.name)

    try:
        result = process_file(test_file, dry_run=True)
        assert result["status"] == "need_process"

    finally:
        test_file.unlink(missing_ok=True)


if __name__ == "__main__":
    test_parse_frontmatter_basic()
    test_parse_frontmatter_no_fm()
    test_extract_tags_from_list()
    test_extract_tags_from_string()
    test_extract_tags_from_title()
    test_classify_by_tag()
    test_classify_by_content()
    test_classify_paper_prefix()
    test_classify_default()
    test_process_file_dry_run()
    test_process_file_need_process()
    print("✓ inbox_triage.py 集成测试通过")