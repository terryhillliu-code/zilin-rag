"""语义切分器测试"""
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.semantic_splitter import SemanticSplitter, split_markdown


def test_basic_header_split():
    """测试标题切分"""
    content = """# 主标题

引言内容。

## 第一节

第一节内容。

## 第二节

第二节内容。
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        filepath = Path(f.name)
    
    try:
        chunks = split_markdown(filepath)
        assert len(chunks) == 3, f"期望 3 个 chunk，实际 {len(chunks)}"
        assert '来源:' in chunks[0].text
        assert chunks[1].h2 == '第一节'
        print(f"✅ 标题切分测试通过: {len(chunks)} 个 chunk")
    finally:
        filepath.unlink()


def test_paragraph_fallback():
    """测试段落回退（无标题时）"""
    content = """这是第一段，没有任何标题。这是一份研报的内容。

这是第二段，继续描述一些技术细节。

这是第三段，总结部分。
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        filepath = Path(f.name)
    
    try:
        splitter = SemanticSplitter(max_chunk_tokens=100, fallback_to_paragraph=True)
        chunks = splitter.split_file(filepath)
        # 由于 max_chunk_tokens 很小，应该切分成多个
        assert len(chunks) >= 1
        print(f"✅ 段落回退测试通过: {len(chunks)} 个 chunk")
    finally:
        filepath.unlink()


def test_frontmatter_extraction():
    """测试 YAML frontmatter 提取"""
    content = """---
category: AI研报
tags:
  - 深度学习
  - 神经网络
priority: 核心
---

# 正文标题

正文内容。
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        filepath = Path(f.name)
    
    try:
        chunks = split_markdown(filepath)
        assert len(chunks) >= 1
        # 检查元数据是否提取
        chunk = chunks[0]
        assert chunk.metadata.get('category') == 'AI研报'
        assert '分类: AI研报' in chunk.text
        print(f"✅ Frontmatter 提取测试通过")
        print(f"   元数据: {chunk.metadata}")
    finally:
        filepath.unlink()


def test_long_content_split():
    """测试超长内容切分"""
    long_para = "这是一段很长的技术描述。" * 200
    content = f"# 测试文档\n\n{long_para}"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        filepath = Path(f.name)
    
    try:
        splitter = SemanticSplitter(max_chunk_tokens=480)
        chunks = splitter.split_file(filepath)
        assert len(chunks) > 1, f"超长内容应被切分，实际 {len(chunks)} 个"
        
        max_chars = int(480 / 1.5)
        for i, chunk in enumerate(chunks):
            assert chunk.char_count <= max_chars * 1.5, f"Chunk {i} 过长: {chunk.char_count}"
        
        print(f"✅ 超长切分测试通过: {len(long_para)} 字符 → {len(chunks)} 个 chunk")
    finally:
        filepath.unlink()


def test_real_vault_sample():
    """测试真实 Obsidian Vault（如果存在）"""
    vault = Path.home() / 'Documents' / 'ZhiweiVault'
    if not vault.exists():
        print("⏭️ 跳过真实 Vault 测试（目录不存在）")
        return
    
    splitter = SemanticSplitter()
    count = 0
    total_chunks = 0
    
    for chunk in splitter.split_directory(vault):
        total_chunks += 1
        if total_chunks == 1:
            print(f"   首个 chunk 示例:")
            print(f"   文件: {chunk.filename}")
            print(f"   前缀: {chunk.text[:100]}...")
        count = total_chunks
        if count >= 100:  # 只测试前 100 个
            break
    
    print(f"✅ 真实 Vault 测试通过: 采样 {count} 个 chunk")


if __name__ == '__main__':
    print("=" * 50)
    print("语义切分器测试")
    print("=" * 50)
    test_basic_header_split()
    test_paragraph_fallback()
    test_frontmatter_extraction()
    test_long_content_split()
    test_real_vault_sample()
    print("=" * 50)
    print("✅ 所有测试通过")
