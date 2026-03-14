"""
多模态处理模块

支持：
- PPT 解析（文字 + 图片提取）
- 图片 OCR（PaddleOCR）
- 图片理解（本地 VLM / 云端降级）
"""

from .ocr_engine import OCREngine
from .vlm_engine import VLMEngine
from .image_extractor import ImageExtractor

__all__ = ["OCREngine", "VLMEngine", "ImageExtractor"]