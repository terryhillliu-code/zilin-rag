"""
Markdown 语义切分器
- 按标题层级切分（适合 Obsidian 笔记）
- 段落回退切分（适合研报转换后的 Markdown）
- 元数据注入到 chunk 开头
"""
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator, Optional


@dataclass
class Chunk:
    """文档片段"""
    text: str                           # 包含元数据前缀的完整文本
    raw_text: str                       # 原始文本
    source: str                         # 文件路径
    filename: str                       # 文件名
    h1: str = ''                        # 一级标题
    h2: str = ''                        # 二级标题
    char_count: int = 0                 # 字符数
    metadata: dict = field(default_factory=dict)  # 额外元数据（来自 frontmatter）


class SemanticSplitter:
    """智能切分器：优先按标题，无标题时按段落"""
    
    def __init__(
        self,
        max_chunk_tokens: int = 480,
        min_chunk_chars: int = 100,
        fallback_to_paragraph: bool = True
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_chars = min_chunk_chars
        self.fallback_to_paragraph = fallback_to_paragraph
        self.max_chunk_chars = int(max_chunk_tokens / 1.5)
    
    def split_file(
        self,
        filepath: Path,
        extra_metadata: Optional[dict] = None
    ) -> list[Chunk]:
        """切分单个文件"""
        if not filepath.exists():
            return []
        
        content = filepath.read_text(encoding='utf-8')
        filename = filepath.stem
        
        # 提取 YAML frontmatter（如果有）
        content, frontmatter = self._extract_frontmatter(content)
        metadata = {**(frontmatter or {}), **(extra_metadata or {})}
        
        # 尝试按标题切分
        chunks = self._split_by_headers(content, filename, str(filepath), metadata)
        
        # 如果只有 1 个 chunk 且超长，说明没有有效标题，回退到段落切分
        if (
            self.fallback_to_paragraph
            and len(chunks) == 1
            and chunks[0].char_count > self.max_chunk_chars
        ):
            chunks = self._split_by_paragraphs(content, filename, str(filepath), metadata)
        
        # 处理超长 chunk
        processed = []
        for chunk in chunks:
            if chunk.char_count > self.max_chunk_chars:
                processed.extend(self._split_long_chunk(chunk))
            elif chunk.char_count < self.min_chunk_chars and processed:
                # 太短则合并到上一个
                prev = processed[-1]
                merged = self._merge_chunks(prev, chunk)
                processed[-1] = merged
            else:
                processed.append(chunk)
        
        return processed
    
    def split_directory(
        self,
        dir_path: Path,
        pattern: str = '**/*.md',
        skip_hidden: bool = True
    ) -> Generator[Chunk, None, None]:
        """切分目录下所有文件"""
        for filepath in dir_path.glob(pattern):
            if skip_hidden and (
                filepath.name.startswith('.')
                or filepath.name.startswith('_')
                or '/.obsidian/' in str(filepath)
            ):
                continue
            for chunk in self.split_file(filepath):
                yield chunk
    
    def _extract_frontmatter(self, content: str) -> tuple[str, Optional[dict]]:
        """提取 YAML frontmatter"""
        if not content.startswith('---'):
            return content, None
        
        end = content.find('---', 3)
        if end < 0:
            return content, None
        
        try:
            import yaml
            frontmatter = yaml.safe_load(content[3:end])
            remaining = content[end + 3:].strip()
            return remaining, frontmatter
        except Exception:
            return content, None
    
    def _split_by_headers(
        self,
        content: str,
        filename: str,
        source: str,
        metadata: dict
    ) -> list[Chunk]:
        """按标题层级切分"""
        chunks = []
        current_h1 = filename
        current_h2 = ''
        current_lines = []
        
        for line in content.split('\n'):
            if line.startswith('# ') and not line.startswith('##'):
                if current_lines:
                    chunks.append(self._build_chunk(
                        '\n'.join(current_lines),
                        source, filename, current_h1, current_h2, metadata
                    ))
                current_h1 = line[2:].strip()
                current_h2 = ''
                current_lines = []
            elif line.startswith('## ') and not line.startswith('###'):
                if current_lines:
                    chunks.append(self._build_chunk(
                        '\n'.join(current_lines),
                        source, filename, current_h1, current_h2, metadata
                    ))
                current_h2 = line[3:].strip()
                current_lines = []
            else:
                current_lines.append(line)
        
        if current_lines:
            chunks.append(self._build_chunk(
                '\n'.join(current_lines),
                source, filename, current_h1, current_h2, metadata
              ))
        
        return chunks
    
    def _split_by_paragraphs(
        self,
        content: str,
        filename: str,
        source: str,
        metadata: dict
    ) -> list[Chunk]:
        """按段落切分（研报回退模式）"""
        paragraphs = re.split(r'\n\s*\n', content)
        
        chunks = []
        current_text = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_len = len(para)
            
            if current_len + para_len > self.max_chunk_chars and current_text:
                chunks.append(self._build_chunk(
                    '\n\n'.join(current_text),
                    source, filename, filename, '', metadata
                ))
                current_text = [para]
                current_len = para_len
            else:
                current_text.append(para)
                current_len += para_len
        
        if current_text:
            chunks.append(self._build_chunk(
                '\n\n'.join(current_text),
                source, filename, filename, '', metadata
            ))
        
        return chunks
    
    def _split_long_chunk(self, chunk: Chunk) -> list[Chunk]:
        """切分超长 chunk"""
        paragraphs = re.split(r'\n\s*\n', chunk.raw_text)
        
        result = []
        current = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if current_len + len(para) > self.max_chunk_chars and current:
                result.append(self._build_chunk(
                    '\n\n'.join(current),
                    chunk.source, chunk.filename, chunk.h1, chunk.h2, chunk.metadata
                ))
                current = [para]
                current_len = len(para)
            else:
                current.append(para)
                current_len += len(para)
        
        if current:
            result.append(self._build_chunk(
                '\n\n'.join(current),
                chunk.source, chunk.filename, chunk.h1, chunk.h2, chunk.metadata
            ))
        
        return result
    
    def _build_chunk(
        self,
        raw_text: str,
        source: str,
        filename: str,
        h1: str,
        h2: str,
        metadata: dict
    ) -> Chunk:
        """构建 Chunk，注入元数据前缀"""
        raw_text = raw_text.strip()
        if not raw_text:
            raw_text = "(空)"
        
        # 构建前缀
        prefix_parts = [f'来源: {filename}']
        if h1 and h1 != filename:
            prefix_parts.append(f'章节: {h1}')
        if h2:
            prefix_parts.append(f'小节: {h2}')
        
        # 添加元数据标签
        if metadata.get('category'):
            prefix_parts.append(f'分类: {metadata["category"]}')
        if metadata.get('tags'):
            tags = metadata['tags']
            if isinstance(tags, list):
                tags = ', '.join(tags[:3])
            prefix_parts.append(f'标签: {tags}')
        
        prefix = ' | '.join(prefix_parts)
        full_text = f'{prefix}\n\n{raw_text}'
        
        return Chunk(
            text=full_text,
            raw_text=raw_text,
            source=source,
            filename=filename,
            h1=h1,
            h2=h2,
            char_count=len(raw_text),
            metadata=metadata
        )
    
    def _merge_chunks(self, chunk1: Chunk, chunk2: Chunk) -> Chunk:
        """合并两个 chunk"""
        merged_raw = chunk1.raw_text + '\n\n' + chunk2.raw_text
        return self._build_chunk(
            merged_raw,
            chunk1.source,
            chunk1.filename,
            chunk1.h1,
            chunk1.h2 or chunk2.h2,
            {**chunk1.metadata, **chunk2.metadata}
        )


# 快捷函数
def split_markdown(filepath: Path, **kwargs) -> list[Chunk]:
    return SemanticSplitter(**kwargs).split_file(filepath)


def split_vault(vault_path: Path, **kwargs) -> Generator[Chunk, None, None]:
    return SemanticSplitter(**kwargs).split_directory(vault_path)
