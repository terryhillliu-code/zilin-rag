"""
Obsidian 摘要自动填充工具 v2
- 扫描待填充文件
- 读取 file:// 链接指向的 PDF
- 调用 LLM 生成结构化知识卡片
- 回填到 MD 文件
"""
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import unquote
from dataclasses import dataclass
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# 加载环境变量
_env_path = Path.home() / "zhiwei-bot" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


@dataclass
class UnfilledFile:
    """待填充文件"""
    md_path: Path
    pdf_path: Optional[Path]
    title: str


@dataclass
class StructuredSummary:
    """结构化摘要"""
    core_contributions: List[str]
    summary: str
    key_concepts: List[str]
    domain: str
    tags: List[str]


class ObsidianSummaryFiller:
    """Obsidian 摘要填充器 v2"""

    PLACEHOLDER = "*(等待向量化及摘要提取后自动写入)*"
    SUMMARY_MODEL = "qwen-turbo"
    SUMMARY_VERSION = "v2"

    def __init__(self, vault_path: str):
        self.vault = Path(vault_path).expanduser()
        self.llm_client = None  # 延迟初始化

    def find_unfilled_files(self, limit: int = 0) -> List[UnfilledFile]:
        """
        找出所有待填充文件

        Args:
            limit: 最大数量，0 表示不限制

        Returns:
            待填充文件列表
        """
        unfilled = []
        unfilled_no_pdf = []

        for md_file in self.vault.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")

                # 检查是否包含占位符
                if self.PLACEHOLDER not in content:
                    continue

                # 提取 PDF 路径
                pdf_path = self._extract_pdf_path(content)

                # 提取标题
                title = self._extract_title(content)

                file_info = UnfilledFile(
                    md_path=md_file,
                    pdf_path=pdf_path,
                    title=title
                )

                # 优先收集有 PDF 且存在的文件
                if pdf_path and pdf_path.exists():
                    unfilled.append(file_info)
                else:
                    unfilled_no_pdf.append(file_info)

                if limit > 0 and len(unfilled) >= limit:
                    break

            except Exception as e:
                print(f"[扫描] 错误: {md_file.name} - {e}")

        # 合并：优先有 PDF 的，再补充没有 PDF 的
        result = unfilled + unfilled_no_pdf
        if limit > 0:
            result = result[:limit]

        return result

    def _extract_pdf_path(self, content: str) -> Optional[Path]:
        """从 MD 内容中提取 PDF 路径"""
        # 匹配 file:// 链接
        match = re.search(r'file://(/[^\)]+)', content)
        if not match:
            return None

        url_path = match.group(1)

        # URL 解码
        decoded = unquote(url_path)

        path = Path(decoded)

        # 只返回 PDF 文件
        if path.suffix.lower() == '.pdf':
            return path

        return None

    def _extract_title(self, content: str) -> str:
        """提取标题"""
        # 从 frontmatter 提取
        match = re.search(r'^title:\s*"([^"]+)"', content, re.MULTILINE)
        if match:
            return match.group(1)

        # 从一级标题提取
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1)

        return "未知标题"

    def _extract_frontmatter_field(self, content: str, field: str) -> str:
        """提取 frontmatter 字段"""
        # 匹配带引号的值
        match = re.search(rf'^{field}:\s*"([^"]*)"', content, re.MULTILINE)
        if match:
            return match.group(1)

        # 匹配不带引号的值
        match = re.search(rf'^{field}:\s*(\S+)', content, re.MULTILINE)
        if match:
            return match.group(1)

        return ""

    def _extract_ingest_date(self, content: str) -> str:
        """提取入库时间"""
        match = re.search(r'入库时间[：:]\s*`?(\d{4}-\d{2}-\d{2})', content)
        if match:
            return match.group(1)
        return ""

    async def generate_summary(self, pdf_path: Path, title: str = "") -> Tuple[Optional[StructuredSummary], str]:
        """
        调用 LLM 生成结构化摘要

        Args:
            pdf_path: PDF 文件路径
            title: 文档标题

        Returns:
            (StructuredSummary, error_msg)
        """
        # 延迟初始化 LLM 客户端
        if self.llm_client is None:
            from openai import AsyncOpenAI
            self.llm_client = AsyncOpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

        # 读取 PDF 内容
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(pdf_path))

            # 提取前几页文字
            text_parts = []
            for page_num in range(min(8, len(doc))):
                page = doc[page_num]
                text_parts.append(page.get_text())
            doc.close()

            full_text = "\n\n".join(text_parts)

            # 截断到前 8000 字符
            if len(full_text) > 8000:
                full_text = full_text[:8000] + "..."

        except Exception as e:
            return None, f"无法读取 PDF: {e}"

        # 调用 LLM - 结构化输出
        prompt = f"""请为这篇文档生成结构化知识卡片，输出严格 JSON 格式：

{{
    "core_contributions": ["贡献1（一句话，加粗关键词）", "贡献2", "贡献3"],
    "summary": "200字内容摘要",
    "key_concepts": ["概念1", "概念2", "概念3"],
    "domain": "所属领域",
    "tags": ["标签1", "标签2", "标签3"]
}}

文档标题：{title}

文档内容：
{full_text}

要求：
1. core_contributions：3 个要点，每个一句话，用 Markdown **加粗** 关键词
2. summary：约 200 字，概述核心内容
3. key_concepts：3-5 个关键术语，用中文，统一规范命名
4. domain：一个主领域（如 deep-learning, finance, infrastructure）
5. tags：3-5 个标签，使用中文
6. 只输出 JSON，不要多余文字"""

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.SUMMARY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content.strip()

            # 解析 JSON
            try:
                data = json.loads(result_text)
            except json.JSONDecodeError:
                # 尝试提取 JSON 块
                json_match = re.search(r'\{[\s\S]*\}', result_text)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    return None, f"JSON 解析失败"

            # 验证字段
            required = ["core_contributions", "summary", "key_concepts", "domain", "tags"]
            for field in required:
                if field not in data:
                    data[field] = [] if field in ["core_contributions", "key_concepts", "tags"] else "未知"

            return StructuredSummary(
                core_contributions=data.get("core_contributions", [])[:5],
                summary=data.get("summary", ""),
                key_concepts=data.get("key_concepts", [])[:5],
                domain=data.get("domain", "general"),
                tags=data.get("tags", [])[:5]
            ), ""

        except Exception as e:
            return None, f"LLM 调用失败: {e}"

    def _generate_doc_id(self, source_path: str) -> str:
        """生成文档 ID (pdf_xxxxxx 格式)"""
        h = hashlib.md5(source_path.encode()).hexdigest()[:6]
        return f"pdf_{h}"

    def _format_contributions(self, contributions: List[str]) -> str:
        """格式化核心贡献"""
        if not contributions:
            return "- 暂无"
        lines = []
        for c in contributions:
            lines.append(f"- {c}")
        return "\n".join(lines)

    def _format_key_concepts(self, concepts: List[str]) -> str:
        """格式化关键概念为 wiki-link 列表"""
        if not concepts:
            return "- 暂无"
        lines = []
        for c in concepts:
            lines.append(f"- [[{c}]]")
        return "\n".join(lines)

    def fill_summary(
        self,
        md_file: Path,
        structured_summary: StructuredSummary,
        title: str,
        author: str = "未知",
        doc_date: str = "",
        doc_type: str = "report",
        ingest_date: str = ""
    ) -> bool:
        """
        将结构化摘要写入 MD 文件

        Args:
            md_file: MD 文件路径
            structured_summary: 结构化摘要
            title: 文档标题
            author: 作者
            doc_date: 文档日期
            doc_type: 文档类型
            ingest_date: 入库日期

        Returns:
            是否成功
        """
        try:
            content = md_file.read_text(encoding="utf-8")

            # 提取原文链接（用于生成 doc_id）
            source_match = re.search(r'\[📄 原文\]\(([^\)]+)\)', content)
            if source_match:
                source_link = source_match.group(1)
                source_path_display = f"[📄 原文]({source_link})"
            else:
                # 从 file:// 提取
                file_match = re.search(r'file://(/[^\)]+)', content)
                if file_match:
                    source_link = f"file://{unquote(file_match.group(1))}"
                    source_path_display = f"[📄 原文]({source_link})"
                else:
                    source_link = str(md_file)
                    source_path_display = "[📄 原文](未找到)"

            # 生成 doc_id（基于 source_path）
            doc_id = self._generate_doc_id(source_link)

            # 格式化标签
            tags_str = ", ".join([f'"{t}"' for t in structured_summary.tags])

            # 构建新 frontmatter（12 字段）
            new_frontmatter = f'''---
title: "{title}"
author: "{author}"
date: {doc_date or datetime.now().strftime("%Y-%m-%d")}
type: {doc_type}
format: pdf
doc_id: "{doc_id}"
source_type: pdf
domain: "{structured_summary.domain}"
summary_model: "{self.SUMMARY_MODEL}"
summary_prompt_version: "{self.SUMMARY_VERSION}"
vectorized: true
summary_status: done
tags: [{tags_str}]
---'''

            # 构建新内容
            new_content = f'''{new_frontmatter}

# {title}

{source_path_display}

> [!info] 文档信息
> 格式：PDF | 入库：{ingest_date or datetime.now().strftime("%Y-%m-%d")}

## 💡 核心贡献

{self._format_contributions(structured_summary.core_contributions)}

## 📝 内容摘要

{structured_summary.summary}

## 🔑 关键概念

{self._format_key_concepts(structured_summary.key_concepts)}

---
> 🤖 由 {self.SUMMARY_MODEL} 生成（prompt {self.SUMMARY_VERSION}）| {datetime.now().strftime("%Y-%m-%d")} | 建议阅读原文后审核
'''

            # 写入文件
            md_file.write_text(new_content, encoding="utf-8")
            return True

        except Exception as e:
            print(f"[填充] 错误: {md_file.name} - {e}")
            return False

    async def process_batch(
        self,
        limit: int = 10,
        dry_run: bool = False,
        skip_missing_pdf: bool = True
    ) -> dict:
        """
        批量处理

        Args:
            limit: 最大处理数量
            dry_run: 只扫描不处理
            skip_missing_pdf: 跳过 PDF 不存在的文件

        Returns:
            处理统计
        """
        print(f"\n[扫描] 查找待填充文件...")
        files = self.find_unfilled_files(limit=limit if limit > 0 else 9999)

        stats = {
            "total_found": len(files),
            "processed": 0,
            "success": 0,
            "skipped_no_pdf": 0,
            "skipped_pdf_missing": 0,
            "failed": 0
        }

        print(f"[扫描] 找到 {len(files)} 个待填充文件")

        if dry_run:
            print("\n[预览] 文件列表:")
            for i, f in enumerate(files[:20]):
                pdf_status = "✅" if f.pdf_path and f.pdf_path.exists() else "❌"
                print(f"  {i+1}. [{pdf_status}] {f.title[:50]}...")
                if f.pdf_path:
                    print(f"      PDF: {f.pdf_path.name}")
            return stats

        # 实际处理
        for i, file in enumerate(files):
            print(f"\n[{i+1}/{len(files)}] 处理: {file.title[:50]}...")

            # 检查 PDF
            if not file.pdf_path:
                print("  跳过: 未找到 PDF 链接")
                stats["skipped_no_pdf"] += 1
                continue

            if not file.pdf_path.exists():
                print(f"  跳过: PDF 不存在 - {file.pdf_path}")
                stats["skipped_pdf_missing"] += 1
                continue

            # 读取现有 frontmatter
            try:
                content = file.md_path.read_text(encoding="utf-8")
                author = self._extract_frontmatter_field(content, "author")
                doc_date = self._extract_frontmatter_field(content, "date")
                doc_type = self._extract_frontmatter_field(content, "type") or "report"
                ingest_date = self._extract_ingest_date(content)
            except:
                author = "未知"
                doc_date = ""
                doc_type = "report"
                ingest_date = ""

            # 生成摘要
            print("  生成结构化摘要...")
            structured_summary, error = await self.generate_summary(file.pdf_path, file.title)

            if error:
                print(f"  失败: {error}")
                stats["failed"] += 1
                continue

            # 写入文件
            print("  写入新格式...")
            if self.fill_summary(
                file.md_path,
                structured_summary,
                title=file.title,
                author=author,
                doc_date=doc_date,
                doc_type=doc_type,
                ingest_date=ingest_date
            ):
                print(f"  ✅ 完成 - 领域: {structured_summary.domain}")
                stats["success"] += 1
            else:
                stats["failed"] += 1

            stats["processed"] += 1

        print(f"\n[完成] 统计:")
        print(f"  找到: {stats['total_found']}")
        print(f"  处理: {stats['processed']}")
        print(f"  成功: {stats['success']}")
        print(f"  跳过(无PDF): {stats['skipped_no_pdf']}")
        print(f"  跳过(PDF不存在): {stats['skipped_pdf_missing']}")
        print(f"  失败: {stats['failed']}")

        return stats


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Obsidian 摘要填充工具")
    parser.add_argument("--vault", default="~/Documents/ZhiweiVault", help="Vault 路径")
    parser.add_argument("--limit", type=int, default=10, help="最大处理数量")
    parser.add_argument("--dry-run", action="store_true", help="只扫描不处理")
    parser.add_argument("--all", action="store_true", help="处理所有文件")

    args = parser.parse_args()

    filler = ObsidianSummaryFiller(args.vault)

    limit = 0 if args.all else args.limit

    await filler.process_batch(
        limit=limit,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    asyncio.run(main())