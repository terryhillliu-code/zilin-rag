"""
图片处理器 - 统一处理图片分析

支持：
- 图片分类（文字/图表/照片）
- OCR 文字识别（PaddleOCR 或 VLM 降级）
- VLM 图像理解
- 自动路由到最佳处理方式

降级策略：
- PaddleOCR 不可用时，使用 VLM 进行文字识别
- 本地 VLM 不可用时，使用云端 API
"""
import os
import sys
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class ImageProcessResult:
    """图片处理结果"""
    image_path: str                    # 图片路径
    image_type: str                    # 图片类型: text/chart/photo/mixed
    ocr_text: str = ""                 # OCR 文字
    vlm_description: str = ""          # VLM 描述
    combined_text: str = ""            # 组合文本（用于向量化）
    confidence: float = 0.0            # 置信度
    processing_time: float = 0.0       # 处理耗时（秒）
    models_used: List[str] = field(default_factory=list)  # 使用的模型


class ImageProcessor:
    """
    图片处理器

    自动判断图片类型，选择最佳处理方式
    """

    # 图片类型判断阈值
    TEXT_DENSITY_THRESHOLD = 0.3      # 文字密度阈值
    CHART_COLOR_THRESHOLD = 50        # 图表颜色数阈值

    def __init__(
        self,
        prefer_local: bool = True,
        ocr_fallback: bool = True,
        vlm_fallback: bool = True,
        api_key: Optional[str] = None
    ):
        """
        初始化图片处理器

        Args:
            prefer_local: 优先使用本地模型
            ocr_fallback: OCR 失败时降级
            vlm_fallback: VLM 失败时降级
            api_key: 云端 API 密钥
        """
        self.prefer_local = prefer_local
        self.ocr_fallback = ocr_fallback
        self.vlm_fallback = vlm_fallback
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")

        self._ocr_engine = None
        self._vlm_engine = None

    def _lazy_init(self):
        """延迟初始化引擎"""
        # 初始化 OCR（如果可用）
        if self._ocr_engine is None:
            try:
                from multimodal.ocr_engine import OCREngine, check_ocr_availability
                # 检查 OCR 是否真的可用
                status = check_ocr_availability()
                if status.get("available"):
                    self._ocr_engine = OCREngine()
                    logger.info("[ImageProcessor] OCR 引擎初始化成功")
                else:
                    logger.warning(f"[ImageProcessor] OCR 不可用: {status.get('error', '未知原因')}，将使用 VLM 进行文字识别")
            except Exception as e:
                logger.warning(f"[ImageProcessor] OCR 引擎初始化失败: {e}，将使用 VLM 进行文字识别")

        # 初始化 VLM
        if self._vlm_engine is None:
            try:
                from multimodal.vlm_engine import VLMEngine
                self._vlm_engine = VLMEngine(
                    prefer_local=self.prefer_local,
                    api_key=self.api_key
                )
            except Exception as e:
                logger.warning(f"[ImageProcessor] VLM 引擎初始化失败: {e}")

    def classify_image(self, image_path: str) -> str:
        """
        简单分类图片类型

        Args:
            image_path: 图片路径

        Returns:
            类型: text/chart/photo/mixed
        """
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(image_path).convert("RGB")
            arr = np.array(img)

            # 分析特征
            # 1. 颜色多样性（图表通常颜色少）
            unique_colors = len(np.unique(arr.reshape(-1, 3), axis=0))

            # 2. 边缘密度（文字和图表边缘多）
            # 简化：使用颜色变化来估计
            gray = np.array(img.convert("L"))
            gradient = np.abs(np.diff(gray.astype(float), axis=0))
            edge_density = np.mean(gradient > 20)

            # 3. 亮度分布
            brightness = np.mean(gray)

            # 判断逻辑
            if edge_density > self.TEXT_DENSITY_THRESHOLD:
                # 边缘多，可能是文字或图表
                if unique_colors < self.CHART_COLOR_THRESHOLD:
                    return "chart"
                else:
                    return "text"
            else:
                # 边缘少，可能是照片
                return "photo"

        except Exception as e:
            logger.warning(f"[ImageProcessor] 图片分类失败: {e}")
            return "mixed"

    def process_image(
        self,
        image_path: str,
        force_type: Optional[str] = None
    ) -> ImageProcessResult:
        """
        处理单张图片

        Args:
            image_path: 图片路径
            force_type: 强制指定类型，跳过自动分类

        Returns:
            ImageProcessResult
        """
        import time

        start_time = time.time()
        self._lazy_init()

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        # 分类
        image_type = force_type or self.classify_image(image_path)

        result = ImageProcessResult(
            image_path=image_path,
            image_type=image_type
        )

        # 根据类型选择处理方式
        if image_type == "text":
            # 文字为主：优先 OCR
            result = self._process_as_text(image_path, result)

        elif image_type == "chart":
            # 图表为主：优先 VLM
            result = self._process_as_chart(image_path, result)

        elif image_type == "photo":
            # 照片：仅 VLM
            result = self._process_as_photo(image_path, result)

        else:  # mixed
            # 混合：OCR + VLM
            result = self._process_as_mixed(image_path, result)

        result.processing_time = time.time() - start_time

        # 生成组合文本
        result.combined_text = self._combine_results(
            result.ocr_text,
            result.vlm_description,
            image_type
        )

        return result

    def _process_as_text(self, image_path: str, result: ImageProcessResult) -> ImageProcessResult:
        """按文字图片处理

        优先使用 PaddleOCR，不可用时降级到 VLM 进行文字识别
        """
        # OCR
        if self._ocr_engine:
            try:
                ocr_result = self._ocr_engine.recognize(image_path)
                result.ocr_text = ocr_result.text
                result.confidence = ocr_result.confidence
                result.models_used.append("PaddleOCR")
            except Exception as e:
                logger.warning(f"[ImageProcessor] OCR 失败: {e}")

        # 如果 OCR 不可用或效果不好，使用 VLM 进行文字识别
        if not result.ocr_text and self._vlm_engine:
            try:
                vlm_result = self._vlm_engine.describe_image(
                    image_path,
                    prompt="请识别并提取图片中的所有文字内容，保持原有格式。"
                )
                # VLM 做文字识别时，结果放入 ocr_text（因为是提取文字）
                result.ocr_text = vlm_result.description
                result.models_used.append(f"VLM-OCR({vlm_result.model})")
            except Exception as e:
                logger.warning(f"[ImageProcessor] VLM 文字识别失败: {e}")

        return result

    def _process_as_chart(self, image_path: str, result: ImageProcessResult) -> ImageProcessResult:
        """按图表处理"""
        # VLM 描述图表
        if self._vlm_engine:
            try:
                vlm_result = self._vlm_engine.describe_image(
                    image_path,
                    prompt="请详细描述这个图表的内容，包括：图表类型、标题、坐标轴、数据趋势、关键数据点。如果是流程图或示意图，请描述其结构和关系。"
                )
                result.vlm_description = vlm_result.description
                result.models_used.append(vlm_result.model)
            except Exception as e:
                logger.warning(f"[ImageProcessor] VLM 处理图表失败: {e}")

        # 尝试 OCR 提取文字标签
        if self._ocr_engine:
            try:
                ocr_result = self._ocr_engine.recognize(image_path)
                if ocr_result.text:
                    result.ocr_text = ocr_result.text
                    result.models_used.append("PaddleOCR")
            except Exception as e:
                logger.warning(f"[ImageProcessor] OCR 处理图表失败: {e}")

        return result

    def _process_as_photo(self, image_path: str, result: ImageProcessResult) -> ImageProcessResult:
        """按照片处理"""
        if self._vlm_engine:
            try:
                vlm_result = self._vlm_engine.describe_image(
                    image_path,
                    prompt="请描述这张图片的内容，包括主要物体、场景、人物、活动等关键信息。"
                )
                result.vlm_description = vlm_result.description
                result.models_used.append(vlm_result.model)
            except Exception as e:
                logger.warning(f"[ImageProcessor] VLM 处理照片失败: {e}")

        return result

    def _process_as_mixed(self, image_path: str, result: ImageProcessResult) -> ImageProcessResult:
        """按混合内容处理"""
        # 同时使用 OCR 和 VLM
        if self._ocr_engine:
            try:
                ocr_result = self._ocr_engine.recognize(image_path)
                result.ocr_text = ocr_result.text
                result.models_used.append("PaddleOCR")
            except Exception as e:
                logger.warning(f"[ImageProcessor] OCR 失败: {e}")

        if self._vlm_engine:
            try:
                vlm_result = self._vlm_engine.describe_image(
                    image_path,
                    prompt="请详细描述这张图片的内容，包括文字、图表、物体等信息。"
                )
                result.vlm_description = vlm_result.description
                result.models_used.append(vlm_result.model)
            except Exception as e:
                logger.warning(f"[ImageProcessor] VLM 失败: {e}")

        return result

    def _combine_results(
        self,
        ocr_text: str,
        vlm_description: str,
        image_type: str
    ) -> str:
        """组合 OCR 和 VLM 结果"""
        parts = []

        if ocr_text:
            parts.append(f"[文字识别]\n{ocr_text}")

        if vlm_description:
            parts.append(f"[图像描述]\n{vlm_description}")

        if not parts:
            return ""

        return "\n\n".join(parts)

    def process_batch(
        self,
        image_paths: List[str],
        parallel: bool = False
    ) -> List[ImageProcessResult]:
        """
        批量处理图片

        Args:
            image_paths: 图片路径列表
            parallel: 是否并行处理

        Returns:
            处理结果列表
        """
        if parallel:
            # TODO: 实现并行处理
            pass

        results = []
        for path in image_paths:
            try:
                result = self.process_image(path)
                results.append(result)
            except Exception as e:
                logger.error(f"[ImageProcessor] 处理失败 {path}: {e}")
                results.append(ImageProcessResult(
                    image_path=path,
                    image_type="error",
                    combined_text=""
                ))

        return results

    def get_status(self) -> Dict[str, Any]:
        """获取处理器状态"""
        self._lazy_init()

        status = {
            "ocr_available": self._ocr_engine is not None,
            "vlm_available": self._vlm_engine is not None,
            "prefer_local": self.prefer_local,
            "cloud_available": bool(self.api_key)
        }

        if self._ocr_engine:
            try:
                from multimodal.ocr_engine import check_ocr_availability
                status["ocr_status"] = check_ocr_availability()
            except Exception:
                pass

        if self._vlm_engine:
            status["vlm_status"] = self._vlm_engine.is_available()

        return status


def process_image(image_path: str, **kwargs) -> ImageProcessResult:
    """
    便捷函数：处理单张图片

    Args:
        image_path: 图片路径
        **kwargs: 传递给 ImageProcessor 的参数

    Returns:
        ImageProcessResult
    """
    processor = ImageProcessor(**kwargs)
    return processor.process_image(image_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="图片处理测试")
    parser.add_argument("image", help="图片路径")
    parser.add_argument("--type", choices=["text", "chart", "photo", "mixed"], help="强制指定类型")
    parser.add_argument("--status", action="store_true", help="查看处理器状态")
    parser.add_argument("--cloud", action="store_true", help="强制使用云端 API")
    args = parser.parse_args()

    processor = ImageProcessor(prefer_local=not args.cloud)

    if args.status:
        status = processor.get_status()
        print(f"处理器状态: {json.dumps(status, indent=2, ensure_ascii=False)}")
    else:
        result = processor.process_image(args.image, force_type=args.type)
        print(f"图片类型: {result.image_type}")
        print(f"处理时间: {result.processing_time:.2f}s")
        print(f"使用模型: {result.models_used}")
        print(f"\n组合结果:\n{result.combined_text[:500]}")