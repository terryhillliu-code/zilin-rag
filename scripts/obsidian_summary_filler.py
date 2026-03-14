"""
Obsidian 摘要自动填充工具
- 扫描待填充文件
- 读取 file:// 链接指向的 PDF
- 调用 LLM 生成摘要
- 回填到 MD 文件
"""
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import unquote
from dataclasses import dataclass

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


class ObsidianSummaryFiller:
    """Obsidian 摘要填充器"""

    PLACEHOLDER = "*(等待向量化及摘要提取后自动写入)*"

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

                unfilled.append(UnfilledFile(
                    md_path=md_file,
                    pdf_path=pdf_path,
                    title=title
                ))

                if limit > 0 and len(unfilled) >= limit:
                    break

            except Exception as e:
                print(f"[扫描] 错误: {md_file.name} - {e}")

        return unfilled

    def _extract_pdf_path(self, content: str) -> Optional[Path]:
        """从 MD 内容中提取 PDF 路径"""
        # 匹配 file:// 链接
        match = re.search(r'file://(/[^\)]+)', content)
        if not match:
            return None

        url_path = match.group(1)

        # URL 解码
        decoded = unquote(url_path)

        return Path(decoded)

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

    async def generate_summary(self, pdf_path: Path) -> str:
        """
        调用 LLM 生成摘要

        Args:
            pdf_path: PDF 文件路径

        Returns:
            摘要文本
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
            for page_num in range(min(5, len(doc))):
                page = doc[page_num]
                text_parts.append(page.get_text())
            doc.close()

            full_text = "\n\n".join(text_parts)

            # 截断到前 5000 字符
            if len(full_text) > 5000:
                full_text = full_text[:5000] + "..."

        except Exception as e:
            return f"*(无法读取 PDF: {e})*"

        # 调用 LLM
        prompt = f"""请为以下文档生成一份简洁的中文摘要（200-300字）。

文档内容：
{full_text}

要求：
1. 概述文档主题和核心内容
2. 提取关键要点（3-5个）
3. 说明文档的价值或应用场景
4. 使用专业、简洁的语言

请直接输出摘要内容，不要添加标题或额外格式。"""

        try:
            response = await self.llm_client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )

            summary = response.choices[0].message.content.strip()
            return summary

        except Exception as e:
            return f"*(LLM 调用失败: {e})*"

    def fill_summary(self, md_file: Path, summary: str) -> bool:
        """
        将摘要写入 MD 文件

        Args:
            md_file: MD 文件路径
            summary: 摘要内容

        Returns:
            是否成功
        """
        try:
            content = md_file.read_text(encoding="utf-8")

            # 替换占位符
            new_content = content.replace(self.PLACEHOLDER, summary)

            if new_content == content:
                print(f"[填充] 未找到占位符: {md_file.name}")
                return False

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

            # 生成摘要
            print("  生成摘要...")
            summary = await self.generate_summary(file.pdf_path)

            if summary.startswith("*("):
                print(f"  失败: {summary}")
                stats["failed"] += 1
                continue

            # 写入文件
            print("  写入摘要...")
            if self.fill_summary(file.md_path, summary):
                print(f"  ✅ 完成")
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