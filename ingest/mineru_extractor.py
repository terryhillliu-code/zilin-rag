"""
MinerU PDF 提取器
- 通过 subprocess 调用独立 venv 中的 MinerU
- 返回结构化结果（文字、表格、图片路径）
- 支持扫描件 PDF 的 OCR 提取
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class MinerUResult:
    """MinerU 提取结果"""
    text: str = ""                 # 提取的文字（Markdown 格式）
    tables: List[str] = field(default_factory=list)  # 表格列表（HTML 格式）
    image_paths: List[str] = field(default_factory=list)  # 提取的图片路径
    is_scanned: bool = False       # 是否为扫描件
    page_count: int = 0            # 页数
    success: bool = False          # 是否成功
    error: str = ""                # 错误信息
    markdown_path: str = ""        # 生成的 Markdown 文件路径


class MinerUExtractor:
    """MinerU PDF 提取器"""

    # MinerU venv 路径
    MINERU_VENV = Path.home() / "zhiwei-rag" / "mineru-venv"
    MINERU_BIN = MINERU_VENV / "bin" / "mineru"

    # 默认输出目录
    DEFAULT_OUTPUT_DIR = Path.home() / "zhiwei-rag" / "mineru_output"

    def __init__(self, use_mps: bool = False, output_dir: Optional[str] = None):
        """
        初始化 MinerU 提取器

        Args:
            use_mps: 是否使用 MPS 加速（macOS Apple Silicon）
                     注意：MPS 模式 (hybrid-auto-engine) 在某些 PDF 上可能不稳定
                     建议默认使用 CPU 模式 (pipeline)，扫描件场景更可靠
            output_dir: 输出目录，默认 ~/zhiwei-rag/mineru_output
        """
        self.use_mps = use_mps
        self.output_dir = Path(output_dir) if output_dir else self.DEFAULT_OUTPUT_DIR
        self._check_environment()

    def _check_environment(self):
        """检查 MinerU 环境是否就绪"""
        if not self.MINERU_BIN.exists():
            raise RuntimeError(
                f"MinerU 未安装: {self.MINERU_BIN}\n"
                "请运行: cd ~/zhiwei-rag && python3.12 -m venv mineru-venv && source mineru-venv/bin/activate && pip install mineru[all]"
            )

    def extract(self, pdf_path: str) -> MinerUResult:
        """
        提取 PDF 内容

        Args:
            pdf_path: PDF 文件路径

        Returns:
            MinerUResult 结构化结果
        """
        if not os.path.exists(pdf_path):
            return MinerUResult(
                text="",
                success=False,
                error=f"PDF 文件不存在: {pdf_path}"
            )

        # 创建输出目录
        timestamp = int(time.time())
        pdf_name = Path(pdf_path).stem
        output_subdir = self.output_dir / f"{pdf_name}_{timestamp}"

        try:
            # 调用 MinerU
            result = self._run_mineru(pdf_path, str(output_subdir))

            if not result.get("success", False):
                return MinerUResult(
                    text="",
                    success=False,
                    error=result.get("error", "MinerU 执行失败")
                )

            # 解析输出
            return self._parse_output(output_subdir, pdf_name)

        except Exception as e:
            return MinerUResult(
                text="",
                success=False,
                error=f"提取过程出错: {str(e)}"
            )

    def _run_mineru(self, pdf_path: str, output_dir: str) -> dict:
        """
        调用 MinerU 命令行

        Args:
            pdf_path: PDF 文件路径
            output_dir: 输出目录

        Returns:
            执行结果字典
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 选择后端
        # pipeline: 纯 CPU
        # hybrid-auto-engine: 自动检测 GPU/MPS
        backend = "hybrid-auto-engine" if self.use_mps else "pipeline"

        cmd = [
            str(self.MINERU_BIN),
            "-p", pdf_path,
            "-o", output_dir,
            "--backend", backend,
            "--output-format", "markdown"
        ]

        print(f"[MinerU] 执行: {' '.join(cmd)}")

        try:
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 分钟超时
            )

            if result.returncode != 0:
                print(f"[MinerU] 错误输出: {result.stderr}")
                return {
                    "success": False,
                    "error": f"MinerU 返回非零: {result.returncode}"
                }

            print(f"[MinerU] 执行成功")
            return {"success": True}

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "执行超时"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _parse_output(self, output_dir: Path, pdf_name: str) -> MinerUResult:
        """
        解析 MinerU 输出

        MinerU 输出结构（pipeline 模式）:
        output_dir/
        ├── {pdf_name}/
        │   └── auto/
        │       ├── {pdf_name}.md      # Markdown 文件
        │       └── images/            # 图片目录
        │           ├── xxx.jpg
        │           └── ...
        """
        result = MinerUResult()

        # 查找输出目录（可能是 {pdf_name} 或 {pdf_name}/auto）
        pdf_output_dir = output_dir / pdf_name
        if not pdf_output_dir.exists():
            # 可能是自动创建的子目录
            for subdir in output_dir.iterdir():
                if subdir.is_dir():
                    pdf_output_dir = subdir
                    break

        if not pdf_output_dir.exists():
            result.error = f"输出目录不存在: {output_dir}"
            return result

        # 检查是否有 auto 子目录（pipeline 模式）
        auto_dir = pdf_output_dir / "auto"
        hybrid_dir = pdf_output_dir / "hybrid_auto"

        if auto_dir.exists():
            content_dir = auto_dir
        elif hybrid_dir.exists():
            content_dir = hybrid_dir
        else:
            content_dir = pdf_output_dir

        # 读取 Markdown 文件
        md_file = content_dir / f"{pdf_name}.md"
        if not md_file.exists():
            # 尝试查找任意 .md 文件
            md_files = list(content_dir.glob("*.md"))
            if md_files:
                md_file = md_files[0]
            else:
                result.error = f"未找到 Markdown 输出文件，搜索目录: {content_dir}"
                return result

        result.markdown_path = str(md_file)

        try:
            with open(md_file, "r", encoding="utf-8") as f:
                result.text = f.read()
        except Exception as e:
            result.error = f"读取 Markdown 失败: {e}"
            return result

        # 提取表格（从 Markdown 中提取 HTML 表格）
        table_pattern = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)
        result.tables = table_pattern.findall(result.text)

        # 查找图片目录
        images_dir = content_dir / "images"
        if images_dir.exists():
            for img_file in images_dir.iterdir():
                if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
                    result.image_paths.append(str(img_file))

        # 判断是否为扫描件
        # 如果提取的文字主要是图片引用，则是扫描件
        text_without_images = re.sub(r'!\[.*?\]\(.*?\)', '', result.text)
        text_without_images = text_without_images.strip()

        # 如果有效文字少于总文字的 20%，认为是扫描件
        if len(text_without_images) < len(result.text) * 0.2 and result.image_paths:
            result.is_scanned = True

        # 计算页数（从图片文件名推断）
        page_nums = set()
        for img_path in result.image_paths:
            match = re.search(r'page[_-]?(\d+)', Path(img_path).name, re.IGNORECASE)
            if match:
                page_nums.add(int(match.group(1)))

        if page_nums:
            result.page_count = max(page_nums)
        else:
            # 从 Markdown 标题推断
            result.page_count = len(re.findall(r'^#+ .*页', result.text, re.MULTILINE))

        result.success = True
        return result

    def cleanup(self):
        """清理临时输出目录"""
        if self.output_dir.exists():
            try:
                shutil.rmtree(self.output_dir)
                print(f"[MinerU] 已清理输出目录: {self.output_dir}")
            except Exception as e:
                print(f"[MinerU] 清理失败: {e}")

    def cleanup_old_outputs(self, max_age_hours: int = 24):
        """
        清理旧的输出目录

        Args:
            max_age_hours: 最大保留时间（小时）
        """
        if not self.output_dir.exists():
            return

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for item in self.output_dir.iterdir():
            if item.is_dir():
                # 从目录名提取时间戳
                match = re.search(r'_(\d{10})$', item.name)
                if match:
                    timestamp = int(match.group(1))
                    if current_time - timestamp > max_age_seconds:
                        try:
                            shutil.rmtree(item)
                            print(f"[MinerU] 清理旧目录: {item.name}")
                        except Exception:
                            pass


# ==================== 测试入口 ====================

def _test():
    """测试 MinerU 提取器"""
    import glob

    # 查找测试 PDF
    search_paths = [
        "/Users/liufang/Documents/Library/**/*.pdf",
        "/Users/liufang/Documents/**/*.pdf",
    ]

    test_pdf = None
    for pattern in search_paths:
        pdfs = glob.glob(os.path.expanduser(pattern), recursive=True)
        if pdfs:
            test_pdf = pdfs[0]
            break

    if not test_pdf:
        print("未找到测试 PDF 文件")
        return

    print(f"测试 PDF: {test_pdf}")

    # 创建提取器
    try:
        extractor = MinerUExtractor(use_mps=True)
    except RuntimeError as e:
        print(f"环境检查失败: {e}")
        return

    # 提取
    print("\n开始提取...")
    result = extractor.extract(test_pdf)

    print(f"\n=== 提取结果 ===")
    print(f"成功: {result.success}")
    print(f"页数: {result.page_count}")
    print(f"扫描件: {result.is_scanned}")
    print(f"文字长度: {len(result.text)} 字符")
    print(f"表格数: {len(result.tables)}")
    print(f"图片数: {len(result.image_paths)}")

    if result.error:
        print(f"错误: {result.error}")

    if result.text:
        print(f"\n--- 文字预览 (前 500 字符) ---")
        print(result.text[:500])

    if result.image_paths:
        print(f"\n--- 图片路径 ---")
        for path in result.image_paths[:5]:
            print(f"  {path}")


if __name__ == "__main__":
    _test()