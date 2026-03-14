"""
PDF 图片提取器
- 使用 PyMuPDF 提取 PDF 中的图片
- 记录页码和位置信息
- 过滤太小的图片（logo、图标）
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF


@dataclass
class ExtractedImage:
    """提取的图片结构"""
    image_bytes: bytes       # 图片二进制数据
    page: int                # 页码（0-indexed）
    bbox: Tuple[int, int, int, int]  # 边界框 (x0, y0, x1, y1)
    width: int               # 原始宽度（像素）
    height: int              # 原始高度（像素）
    image_index: int         # 页内图片索引
    ext: str                 # 图片格式扩展名


def extract_images(
    pdf_path: str,
    min_size: int = 100,
    min_area: int = 10000
) -> List[ExtractedImage]:
    """
    提取 PDF 中的图片，过滤小图片

    Args:
        pdf_path: PDF 文件路径
        min_size: 最小边长（像素），默认 100
        min_area: 最小面积（平方像素），默认 10000 (100x100)

    Returns:
        提取的图片列表
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    images: List[ExtractedImage] = []

    try:
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                # img_info: (xref, smask, width, height, bpc, colorspace, ...)
                xref = img_info[0]
                base_image = doc.extract_image(xref)

                if base_image is None:
                    continue

                width = base_image["width"]
                height = base_image["height"]
                image_bytes = base_image["image"]
                ext = base_image["ext"]

                # 过滤太小的图片
                if width < min_size or height < min_size:
                    continue

                if width * height < min_area:
                    continue

                # 获取图片在页面上的位置
                # 注意：一个图片可能在页面上出现多次，这里取第一个
                bbox = _get_image_bbox(page, xref)

                images.append(ExtractedImage(
                    image_bytes=image_bytes,
                    page=page_num,
                    bbox=bbox,
                    width=width,
                    height=height,
                    image_index=img_index,
                    ext=ext
                ))

        doc.close()

    except Exception as e:
        print(f"[PDF图片提取] 错误: {e}")

    return images


def _get_image_bbox(page: fitz.Page, xref: int) -> Tuple[int, int, int, int]:
    """
    获取图片在页面上的边界框

    Args:
        page: PyMuPDF 页面对象
        xref: 图片的 xref

    Returns:
        边界框 (x0, y0, x1, y1)，如果找不到返回 (0, 0, 0, 0)
    """
    try:
        # 获取页面上所有图片的位置信息
        img_rects = page.get_image_rects(xref)
        if img_rects:
            # 取第一个出现位置
            rect = img_rects[0]
            return (int(rect.x0), int(rect.y0), int(rect.x1), int(rect.y1))
    except Exception:
        pass

    return (0, 0, 0, 0)


def save_images(
    images: List[ExtractedImage],
    output_dir: str,
    prefix: str = "img"
) -> List[str]:
    """
    将提取的图片保存到文件

    Args:
        images: 图片列表
        output_dir: 输出目录
        prefix: 文件名前缀

    Returns:
        保存的文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for img in images:
        filename = f"{prefix}_p{img.page:03d}_{img.image_index:03d}.{img.ext}"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "wb") as f:
            f.write(img.image_bytes)

        saved_paths.append(filepath)

    return saved_paths


def extract_and_save(
    pdf_path: str,
    output_dir: str,
    min_size: int = 100,
    min_area: int = 10000
) -> Tuple[List[ExtractedImage], List[str]]:
    """
    提取并保存 PDF 图片

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录
        min_size: 最小边长
        min_area: 最小面积

    Returns:
        (图片列表, 保存路径列表)
    """
    images = extract_images(pdf_path, min_size, min_area)

    pdf_name = Path(pdf_path).stem
    saved_paths = save_images(images, output_dir, prefix=pdf_name)

    return images, saved_paths


# ==================== 测试入口 ====================

if __name__ == "__main__":
    import sys
    import glob

    # 查找测试 PDF
    search_paths = [
        "/Users/liufang/Documents/**/*.pdf",
        "~/Documents/**/*.pdf",
    ]

    test_pdf = None
    for pattern in search_paths:
        pdfs = glob.glob(os.path.expanduser(pattern), recursive=True)
        if pdfs:
            test_pdf = pdfs[0]
            break

    if not test_pdf:
        print("未找到测试 PDF 文件")
        sys.exit(1)

    print(f"测试 PDF: {test_pdf}")

    # 提取图片
    images = extract_images(test_pdf)
    print(f"\n提取到 {len(images)} 张图片:")

    for img in images[:10]:
        print(f"  页码: {img.page}, 尺寸: {img.width}x{img.height}, 格式: {img.ext}")
        if img.bbox != (0, 0, 0, 0):
            print(f"    位置: {img.bbox}")

    if len(images) > 10:
        print(f"  ... 还有 {len(images) - 10} 张")