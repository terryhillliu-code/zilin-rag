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
        min_chunk_chars: int = 10,
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
        
        # 处理超长或超短 chunk
        processed = []
        for chunk in chunks:
            if chunk.char_count > self.max_chunk_chars:
                processed.extend(self._split_long_chunk(chunk))
            elif chunk.char_count < self.min_chunk_chars and processed:
                # 只有在标题相同的情况下才合并，或者是段落模式
                prev = processed[-1]
                if prev.h1 == chunk.h1 and prev.h2 == chunk.h2:
                    merged = self._merge_chunks(prev, chunk)
                    processed[-1] = merged
                else:
                    processed.append(chunk)
            else:
                processed.append(chunk)
        
        return processed
    
    def split_directory(
        self,
        dir_path: Path,
        pattern: str = '**/*.md',
        skip_hidden: bool = True,
        skip_dirs: Optional[list[str]] = None
    ) -> Generator[Chunk, None, None]:
        """切分目录下所有文件

        Args:
            dir_path: 目录路径
            pattern: 文件匹配模式
            skip_hidden: 跳过隐藏文件
            skip_dirs: 要跳过的目录名列表（如归档目录）

        DKI (Dynamic Knowledge Isolation) 默认隔离目录:
            - Inbox: 收集箱，待处理内容不入库
            - 72_视频笔记_Video-Distill: 视频文稿，非正式知识不入库
            - 90-99_系统与归档_System: 归档内容不入库
            - 92_归档备份: 历史归档不入库
            - attachments: 附件目录不入库
        """
        if skip_dirs is None:
            # DKI 默认隔离列表 - 这些目录的内容不入 RAG 索引
            skip_dirs = [
                'Inbox',                        # 收集箱：待处理内容
                '72_视频笔记_Video-Distill',     # DKI：视频文稿隔离
                '90-99_系统与归档_System',       # 系统归档
                '92_归档备份',                   # 历史归档
                '归档',
                'Archive',
                'attachments',                  # 附件目录
            ]

        for filepath in dir_path.glob(pattern):
            # 检查是否在跳过目录中
            path_str = str(filepath)
            if any(skip_dir in path_str for skip_dir in skip_dirs):
                continue
            if skip_hidden and (
                filepath.name.startswith('.')
                or filepath.name.startswith('_')
                or '/.obsidian/' in path_str
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
        
        lines = content.split('\n')
        for line in lines:
            h1_match = line.startswith('# ') and not line.startswith('##')
            h2_match = line.startswith('## ') and not line.startswith('###')
            
            if h1_match or h2_match:
                # 碰到新标题，如果当前有内容，先存入旧 chunk
                if any(ln.strip() for ln in current_lines):
                    chunks.append(self._build_chunk(
                        '\n'.join(current_lines),
                        source, filename, current_h1, current_h2, metadata
                    ))
                
                if h1_match:
                    current_h1 = line[2:].strip()
                    current_h2 = ''
                else:
                    current_h2 = line[3:].strip()
                current_lines = []
            else:
                current_lines.append(line)
        
        # 处理最后一段内容
        if any(ln.strip() for ln in current_lines):
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
        # 1. 先尝试按段落切
        paragraphs = re.split(r'\n\s*\n', chunk.raw_text)
        
        result = []
        current_text_segments = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 如果单段就超长，递归切分单段
            if len(para) > self.max_chunk_chars:
                # 如果当前有积压，先清空
                if current_text_segments:
                    result.append(self._build_chunk(
                        '\n\n'.join(current_text_segments),
                        chunk.source, chunk.filename, chunk.h1, chunk.h2, chunk.metadata
                    ))
                    current_text_segments = []
                    current_len = 0
                
                # 切分超长单段
                sub_chunks = self._split_single_paragraph(para)
                for sub in sub_chunks:
                    result.append(self._build_chunk(
                        sub,
                        chunk.source, chunk.filename, chunk.h1, chunk.h2, chunk.metadata
                    ))
                continue

            if current_len + len(para) > self.max_chunk_chars and current_text_segments:
                result.append(self._build_chunk(
                    '\n\n'.join(current_text_segments),
                    chunk.source, chunk.filename, chunk.h1, chunk.h2, chunk.metadata
                ))
                current_text_segments = [para]
                current_len = len(para)
            else:
                current_text_segments.append(para)
                current_len += len(para)
        
        if current_text_segments:
            result.append(self._build_chunk(
                '\n\n'.join(current_text_segments),
                chunk.source, chunk.filename, chunk.h1, chunk.h2, chunk.metadata
            ))
        
        return result

    def _split_single_paragraph(self, text: str) -> list[str]:
        """即使没有段落，也强制按字数切分"""
        res = []
        start = 0
        while start < len(text):
            end = start + self.max_chunk_chars
            # 尽量在标点符号处断开
            if end < len(text):
                # 在 end 附近往回找标点
                search_range = text[max(start, end-20):end]
                last_punct = -1
                for p in ["。", "！", "？", ".", "!", "?", "；", ";"]:
                    pos = search_range.rfind(p)
                    if pos > last_punct:
                        last_punct = pos
                
                if last_punct != -1:
                    end = max(start, end-20) + last_punct + 1
            
            res.append(text[start:end].strip())
            start = end
        return res
    
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
