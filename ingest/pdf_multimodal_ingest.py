"""
多模态 PDF 入库
- 提取文字内容 → chunk_type="text"
- 提取图片 → VLM 描述 → chunk_type="figure"
- 统一入库到 LanceDB
- 支持 MinerU OCR（扫描件场景）
"""
import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

import fitz  # PyMuPDF

from .pdf_image_extractor import ExtractedImage, extract_images
from .vlm_describer import VLMDescriber, ImageDescription
from .lance_store import LanceStore, Document


@dataclass
class TextChunk:
    """文字分块"""
    text: str
    page: int
    char_count: int


def extract_text_from_pdf(
    pdf_path: str,
    use_mineru: bool = False,
    mineru_images: Optional[List[str]] = None
) -> Tuple[List[TextChunk], List[str]]:
    """
    从 PDF 提取文字，按页分块

    Args:
        pdf_path: PDF 文件路径
        use_mineru: 是否优先使用 MinerU（扫描件场景）
        mineru_images: MinerU 提取的图片路径列表（输出参数）

    Returns:
        (文字分块列表, 图片路径列表)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    chunks = []
    image_paths = []

    # ========== 优先使用 MinerU（针对扫描件）==========
    if use_mineru:
        try:
            from .mineru_extractor import MinerUExtractor, MinerUResult

            print("  [MinerU] 尝试 OCR 提取...")
            extractor = MinerUExtractor(use_mps=False)
            result = extractor.extract(pdf_path)

            if result.success and result.text.strip():
                print(f"  [MinerU] 提取成功: {len(result.text)} 字符, {len(result.image_paths)} 张图片")

                # MinerU 返回 Markdown，按页面分隔符分割
                # 常见分隔符: "--- Page X ---" 或类似
                pages = _split_mineru_pages(result.text)

                for i, page_text in enumerate(pages):
                    text = _clean_text(page_text)
                    if text.strip():
                        chunks.append(TextChunk(
                            text=text,
                            page=i,
                            char_count=len(text)
                        ))

                # 返回 MinerU 提取的图片路径
                if mineru_images is not None:
                    mineru_images.extend(result.image_paths)
                image_paths = result.image_paths

                # 如果提取到有效文字，直接返回
                if chunks:
                    return chunks, image_paths

            else:
                print(f"  [MinerU] 提取失败或无文字，回退到 PyMuPDF: {result.error}")

        except ImportError:
            print("  [MinerU] 未安装，回退到 PyMuPDF")
        except Exception as e:
            print(f"  [MinerU] 错误: {e}，回退到 PyMuPDF")

    # ========== 回退到 PyMuPDF（现有逻辑）==========
    try:
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            # 清理文本
            text = _clean_text(text)

            if text.strip():
                chunks.append(TextChunk(
                    text=text,
                    page=page_num,
                    char_count=len(text)
                ))

        doc.close()

    except Exception as e:
        print(f"[PDF文字提取] 错误: {e}")

    return chunks, image_paths


def _split_mineru_pages(markdown_text: str) -> List[str]:
    """
    将 MinerU 输出的 Markdown 按页分割

    Args:
        markdown_text: MinerU 输出的 Markdown 文本

    Returns:
        按页分割的文本列表
    """
    # MinerU 常见的页面分隔符
    # 格式: "--- Page N ---" 或 "--- 第 N 页 ---" 或连续的 "## Page N"
    separators = [
        r'\n---+\s*Page\s*\d+\s*---+\n',  # --- Page 1 ---
        r'\n---+\s*第\s*\d+\s*页\s*---+\n',  # --- 第 1 页 ---
        r'\n##\s*Page\s*\d+\s*\n',  # ## Page 1
        r'\n##\s*第\s*\d+\s*页\s*\n',  # ## 第 1 页
        r'\n\\\\newpage\n',  # LaTeX 风格
    ]

    # 尝试按分隔符分割
    for sep in separators:
        parts = re.split(sep, markdown_text, flags=re.IGNORECASE)
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]

    # 无分隔符，返回整体作为一个页面
    return [markdown_text] if markdown_text.strip() else []


def _clean_text(text: str) -> str:
    """清理提取的文本"""
    # 移除多余空白
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def split_large_chunk(
    chunk: TextChunk,
    max_chars: int = 2000,
    overlap: int = 200
) -> List[TextChunk]:
    """
    将大分块切分为小块

    Args:
        chunk: 原始分块
        max_chars: 最大字符数
        overlap: 重叠字符数

    Returns:
        切分后的分块列表
    """
    if chunk.char_count <= max_chars:
        return [chunk]

    chunks = []
    text = chunk.text
    start = 0

    while start < len(text):
        end = start + max_chars

        # 尝试在句子边界切分
        if end < len(text):
            # 找最后一个句号、问号、感叹号
            last_sentence_end = max(
                text.rfind('。', start, end),
                text.rfind('？', start, end),
                text.rfind('！', start, end),
                text.rfind('.', start, end),
            )
            if last_sentence_end > start + max_chars // 2:
                end = last_sentence_end + 1

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(TextChunk(
                text=chunk_text,
                page=chunk.page,
                char_count=len(chunk_text)
            ))

        # 下一块的起点（考虑重叠）
        start = end - overlap if end < len(text) else end

    return chunks


async def ingest_pdf_multimodal(
    pdf_path: str,
    store: LanceStore,
    vlm_describer: Optional[VLMDescriber] = None,
    max_text_chars: int = 2000,
    max_images: int = 50,
    show_progress: bool = True,
    use_mineru: bool = False
) -> dict:
    """
    多模态 PDF 入库

    Args:
        pdf_path: PDF 文件路径
        store: LanceDB 存储
        vlm_describer: VLM 描述器（可选，不传则跳过图片）
        max_text_chars: 文字分块最大字符数
        max_images: 最大处理图片数
        show_progress: 显示进度
        use_mineru: 是否使用 MinerU OCR（扫描件场景）

    Returns:
        {"text_chunks": N, "figure_chunks": M, "total": N+M}
    """
    filename = Path(pdf_path).name

    print(f"\n[多模态入库] 处理: {filename}")
    if use_mineru:
        print("  [模式] MinerU OCR（扫描件优化）")

    # ========== 1. 提取文字 ==========
    print("  [1/3] 提取文字...")
    mineru_images: List[str] = []
    text_chunks, extracted_image_paths = extract_text_from_pdf(
        pdf_path,
        use_mineru=use_mineru,
        mineru_images=mineru_images
    )

    # 切分大块
    all_text_chunks = []
    for chunk in text_chunks:
        split_chunks = split_large_chunk(chunk, max_chars=max_text_chars)
        all_text_chunks.extend(split_chunks)

    print(f"        文字分块: {len(all_text_chunks)} 个")

    # ========== 2. 提取图片并描述 ==========
    figure_descriptions = []

    if vlm_describer:
        print("  [2/3] 提取图片...")

        # 优先使用 MinerU 提取的图片
        if use_mineru and mineru_images:
            print(f"        使用 MinerU 提取的图片: {len(mineru_images)} 张")
            # 从文件路径读取图片并描述
            figure_descriptions = await vlm_describer.describe_batch_from_paths(
                mineru_images[:max_images],
                source=filename,
                show_progress=show_progress
            )
        else:
            # 回退到 PyMuPDF 提取图片
            images = extract_images(pdf_path)

            # 限制数量
            if len(images) > max_images:
                print(f"        ⚠️ 图片过多 ({len(images)})，仅处理前 {max_images} 张")
                images = images[:max_images]

            # 过滤有效图片
            valid_images = []
            for img in images:
                should, reason = vlm_describer.should_process(img)
                if should:
                    valid_images.append(img)

            print(f"        有效图片: {len(valid_images)} 张")

            if valid_images:
                print("  [3/3] VLM 描述图片...")
                figure_descriptions = await vlm_describer.describe_batch(
                    valid_images,
                    source=filename,
                    show_progress=show_progress
                )
    else:
        print("  [2/3] 跳过图片（未提供 VLM 描述器）")

    # ========== 3. 构建 Document 对象 ==========
    documents = []

    # 文字 chunks
    for i, chunk in enumerate(all_text_chunks):
        doc_id = f"{pdf_path}#text_{i}"

        doc = Document(
            id=doc_id,
            text=f"[{filename} 第{chunk.page + 1}页]\n{chunk.text}",
            raw_text=chunk.text,
            source=pdf_path,
            filename=filename,
            h1="",
            h2="",
            category="",
            tags="",
            char_count=chunk.char_count,
            chunk_type="text",
            page=chunk.page,
            timestamp="",
            figure_path=""
        )
        documents.append(doc)

    # 图表 chunks
    for i, desc in enumerate(figure_descriptions):
        if desc.skipped:
            continue

        doc_id = f"{pdf_path}#figure_{i}"

        doc = Document(
            id=doc_id,
            text=f"[{filename} 第{desc.page + 1}页 图表]\n{desc.description}",
            raw_text=desc.description,
            source=pdf_path,
            filename=filename,
            h1="",
            h2="",
            category="",
            tags=f"figure,{desc.image_type}",
            char_count=len(desc.description),
            chunk_type="figure",
            page=desc.page,
            timestamp="",
            figure_path=f"page_{desc.page}_fig_{i}"
        )
        documents.append(doc)

    # ========== 4. 生成向量并入库 ==========
    if documents:
        print(f"  [入库] 生成向量...")
        vectors = _get_embeddings([d.raw_text for d in documents])

        for doc, vec in zip(documents, vectors):
            doc.vector = vec

        store.add_documents(documents)

    # 统计
    text_count = sum(1 for d in documents if d.chunk_type == "text")
    figure_count = sum(1 for d in documents if d.chunk_type == "figure")

    print(f"  [完成] 文字: {text_count}, 图表: {figure_count}, 总计: {len(documents)}")

    return {
        "text_chunks": text_count,
        "figure_chunks": figure_count,
        "total": len(documents)
    }


def _get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    获取文本向量（调用常驻服务）

    Args:
        texts: 文本列表

    Returns:
        向量列表
    """
    import requests

    try:
        resp = requests.post(
            "http://127.0.0.1:8765/embed",
            json={"texts": texts},
            timeout=120
        )

        if resp.status_code == 200:
            return resp.json().get("embeddings", [])
    except Exception as e:
        print(f"[向量] 服务调用失败: {e}")

    # 降级：返回空向量
    return [[] for _ in texts]


# ==================== 批量入库 ====================

async def ingest_pdf_directory(
    dir_path: str,
    store: LanceStore,
    pattern: str = "*.pdf",
    max_files: int = 100,
    skip_existing: bool = True
) -> dict:
    """
    批量入库目录下的 PDF

    Args:
        dir_path: 目录路径
        store: LanceDB 存储
        pattern: 文件匹配模式
        max_files: 最大处理文件数
        skip_existing: 跳过已入库的文件

    Returns:
        {"files": N, "text_chunks": M, "figure_chunks": K, "total": M+K}
    """
    from glob import glob

    # 查找 PDF 文件
    pdf_files = glob(os.path.join(dir_path, "**", pattern), recursive=True)

    if len(pdf_files) > max_files:
        print(f"[批量入库] 文件过多 ({len(pdf_files)})，仅处理前 {max_files} 个")
        pdf_files = pdf_files[:max_files]

    print(f"[批量入库] 找到 {len(pdf_files)} 个 PDF 文件")

    # 初始化 VLM 描述器
    vlm_describer = VLMDescriber(timeout=60)

    # 统计
    total_stats = {
        "files": 0,
        "text_chunks": 0,
        "figure_chunks": 0,
        "total": 0
    }

    for pdf_path in tqdm(pdf_files, desc="入库中"):
        try:
            # 检查是否已入库
            if skip_existing:
                existing = store.search_text(
                    Path(pdf_path).name,
                    top_k=1,
                    filter_sql=f"source = '{pdf_path}'"
                )
                if existing:
                    print(f"  跳过（已存在）: {Path(pdf_path).name}")
                    continue

            # 入库
            result = await ingest_pdf_multimodal(
                pdf_path,
                store,
                vlm_describer,
                show_progress=False
            )

            total_stats["files"] += 1
            total_stats["text_chunks"] += result["text_chunks"]
            total_stats["figure_chunks"] += result["figure_chunks"]
            total_stats["total"] += result["total"]

        except Exception as e:
            print(f"  错误: {Path(pdf_path).name} - {e}")

    print(f"\n[批量入库完成]")
    print(f"  文件数: {total_stats['files']}")
    print(f"  文字分块: {total_stats['text_chunks']}")
    print(f"  图表分块: {total_stats['figure_chunks']}")
    print(f"  总计: {total_stats['total']}")

    return total_stats


# ==================== 测试入口 ====================

async def _test():
    """测试多模态入库"""
    import glob

    # 查找测试 PDF
    search_paths = [
        "/Users/liufang/Documents/clawdbot download/*.pdf",
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

    # 初始化
    store = LanceStore()
    vlm = VLMDescriber(timeout=60)

    # 入库
    result = await ingest_pdf_multimodal(test_pdf, store, vlm)
    print(f"\n入库结果: {result}")

    # 验证
    import lancedb
    db = lancedb.connect(store.db_path)
    table = db.open_table("documents")

    # 查询 figure 类型
    try:
        figures = table.search().where(f'chunk_type = "figure" AND source = "{test_pdf}"').limit(5).to_list()
        print(f"\n图表记录数: {len(figures)}")
        if figures:
            print(f"示例: {figures[0]['content'][:100]}...")
    except Exception as e:
        print(f"查询失败: {e}")


if __name__ == "__main__":
    asyncio.run(_test())