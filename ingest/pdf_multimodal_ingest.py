"""
多模态 PDF 入库 (v5.0 架构整合版)
- 提取文字内容 → chunk_type="text"
- 提取图片 → VLM 描述 (支持 JSON 结构化提取) → chunk_type="figure"
- 统一入库到 LanceDB
- 支持 MinerU OCR（扫描件场景）
"""
import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
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
    """从 PDF 提取文字，按页分块"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    chunks = []
    image_paths = []

    if use_mineru:
        try:
            from .mineru_extractor import MinerUExtractor, MinerUResult
            print("  [MinerU] 尝试 OCR 提取...")
            extractor = MinerUExtractor(use_mps=False)
            result = extractor.extract(pdf_path)

            if result.success and result.text.strip():
                print(f"  [MinerU] 提取成功: {len(result.text)} 字符, {len(result.image_paths)} 张图片")
                pages = _split_mineru_pages(result.text)
                for i, page_text in enumerate(pages):
                    text = _clean_text(page_text)
                    if text.strip():
                        chunks.append(TextChunk(text=text, page=i, char_count=len(text)))
                if mineru_images is not None:
                    mineru_images.extend(result.image_paths)
                image_paths = result.image_paths
                if chunks:
                    return chunks, image_paths
        except Exception as e:
            print(f"  [MinerU] 错误: {e}，回退到 PyMuPDF")

    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = _clean_text(page.get_text())
            if text.strip():
                chunks.append(TextChunk(text=text, page=page_num, char_count=len(text)))
        doc.close()
    except Exception as e:
        print(f"[PDF文字提取] 错误: {e}")

    return chunks, image_paths

def _split_mineru_pages(markdown_text: str) -> List[str]:
    separators = [
        r'\n---+\s*Page\s*\d+\s*---+\n',
        r'\n---+\s*第\s*\d+\s*页\s*---+\n',
        r'\n##\s*Page\s*\d+\s*\n',
        r'\n##\s*第\s*\d+\s*页\s*\n',
        r'\n\\\\newpage\n',
    ]
    for sep in separators:
        parts = re.split(sep, markdown_text, flags=re.IGNORECASE)
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]
    return [markdown_text] if markdown_text.strip() else []

def _clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def split_large_chunk(chunk: TextChunk, max_chars: int = 2000, overlap: int = 200) -> List[TextChunk]:
    if chunk.char_count <= max_chars:
        return [chunk]
    chunks = []
    text = chunk.text
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            last_sentence_end = max(text.rfind('。', start, end), text.rfind('？', start, end), 
                                   text.rfind('！', start, end), text.rfind('.', start, end))
            if last_sentence_end > start + max_chars // 2:
                end = last_sentence_end + 1
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(TextChunk(text=chunk_text, page=chunk.page, char_count=len(chunk_text)))
        start = end - overlap if end < len(text) else end
    return chunks

async def ingest_pdf_multimodal(
    pdf_path: str,
    store: LanceStore,
    vlm_describer: Optional[VLMDescriber] = None,
    max_text_chars: int = 2000,
    max_images: int = 50,
    show_progress: bool = True,
    use_mineru: bool = False,
    vlm_score_threshold: int = 0
) -> dict:
    """多模态 PDF 入库 (Research V5.0)"""
    filename = Path(pdf_path).name
    print(f"\n[多模态入库] 处理: {filename}")

    # 1. 提取文字
    mineru_images: List[str] = []
    text_chunks, _ = extract_text_from_pdf(pdf_path, use_mineru=use_mineru, mineru_images=mineru_images)
    all_text_chunks = []
    for chunk in text_chunks:
        all_text_chunks.extend(split_large_chunk(chunk, max_chars=max_text_chars))

    # 2. 提取图片并描述
    figure_descriptions = []
    if vlm_describer:
        print("  [2/3] 提取图片与 VLM 驱动分析...")
        if use_mineru and mineru_images:
            figure_descriptions = await vlm_describer.describe_batch_from_paths(
                mineru_images[:max_images], source=filename, show_progress=show_progress
            )
        else:
            images = extract_images(pdf_path)[:max_images]
            valid_images = [img for img in images if vlm_describer.should_process(img)[0]]
            if valid_images:
                figure_descriptions = await vlm_describer.describe_batch(
                    valid_images, source=filename, show_progress=show_progress
                )

    # 3. 构建 Document
    documents = []
    for i, chunk in enumerate(all_text_chunks):
        doc = Document(
            id=f"{pdf_path}#text_{i}",
            text=f"[{filename} 第{chunk.page + 1}页]\n{chunk.text}",
            raw_text=chunk.text,
            source=pdf_path,
            filename=filename,
            char_count=chunk.char_count,
            chunk_type="text",
            page=chunk.page
        )
        documents.append(doc)

    for i, desc in enumerate(figure_descriptions):
        if desc.skipped: continue
        doc = Document(
            id=f"{pdf_path}#figure_{i}",
            text=f"[{filename} 第{desc.page + 1}页 图表]\n{desc.description}",
            raw_text=desc.description,
            source=pdf_path,
            filename=filename,
            tags=f"figure,{desc.image_type}",
            char_count=len(desc.description),
            chunk_type="figure",
            page=desc.page,
            figure_path=desc.structured_data.get("title", f"fig_{i}") if desc.structured_data else f"fig_{i}"
        )
        if desc.structured_data:
            doc.metadata["structured_vlm"] = desc.structured_data
        documents.append(doc)

    # 4. 向量化并入库
    if documents:
        from .pdf_multimodal_ingest import _get_embeddings
        vectors = _get_embeddings([d.raw_text for d in documents])
        for doc, vec in zip(documents, vectors):
            doc.vector = vec
        store.add_documents(documents)

    text_count = sum(1 for d in documents if d.chunk_type == "text")
    figure_count = sum(1 for d in documents if d.chunk_type == "figure")
    return {"text_chunks": text_count, "figure_chunks": figure_count, "total": len(documents)}

def _get_embeddings(texts: List[str]) -> List[List[float]]:
    import requests
    try:
        resp = requests.post("http://127.0.0.1:8765/embed", json={"texts": texts}, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("embeddings", [])
    except Exception as e:
        print(f"[向量] 服务调用失败: {e}")
    return [[] for _ in texts]

async def _test():
    store = LanceStore()
    vlm = VLMDescriber(timeout=60)
    import glob
    test_pdf = glob.glob("/Users/liufang/Documents/**/*.pdf", recursive=True)[0]
    result = await ingest_pdf_multimodal(test_pdf, store, vlm)
    print(f"入库结果: {result}")

if __name__ == "__main__":
    asyncio.run(_test())