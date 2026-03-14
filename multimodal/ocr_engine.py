"""
OCR 引擎 - PaddleOCR 封装

本地优先，支持：
- 中英文识别
- 倾斜文字检测
- macOS MPS 加速（如果支持）
"""
import os
import sys
from typing import Optional, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """OCR 结果"""
    text: str                    # 识别文本
    confidence: float            # 置信度
    boxes: List[List[int]] = None  # 文字框坐标


class OCREngine:
    """
    PaddleOCR 引擎封装

    使用本地 PaddleOCR 进行文字识别，优先使用 MPS/GPU 加速
    """

    def __init__(
        self,
        use_gpu: bool = False,
        use_mps: bool = True,
        lang: str = "ch",
        use_angle_cls: bool = True
    ):
        """
        初始化 OCR 引擎

        Args:
            use_gpu: 是否使用 GPU
            use_mps: 是否使用 macOS MPS 加速
            lang: 语言，'ch' 表示中英文，'en' 表示英文
            use_angle_cls: 是否检测文字方向
        """
        self.use_gpu = use_gpu
        self.use_mps = use_mps
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self._ocr = None
        self._initialized = False

    def _lazy_init(self):
        """延迟初始化，首次使用时才加载模型"""
        if self._initialized:
            return

        try:
            from paddleocr import PaddleOCR

            # 确定设备
            device = self._detect_device()

            # 创建 OCR 实例
            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=device == "gpu",
                enable_mkldnn=device == "cpu",  # CPU 优化
                show_log=False,
                # 禁用自动下载日志
                disable_mkldnn=False if device == "cpu" else True
            )

            self._initialized = True
            logger.info(f"[OCREngine] 初始化成功，设备: {device}")

        except ImportError as e:
            logger.error(f"[OCREngine] PaddleOCR 未安装: {e}")
            logger.error("请运行: pip install paddlepaddle paddleocr")
            raise RuntimeError("PaddleOCR 未安装，请先安装依赖")
        except Exception as e:
            logger.error(f"[OCREngine] 初始化失败: {e}")
            raise

    def _detect_device(self) -> str:
        """检测可用设备"""
        # 检查 GPU
        if self.use_gpu:
            try:
                import paddle
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.set_device("gpu")
                    return "gpu"
            except Exception:
                pass

        # 检查 MPS (Apple Silicon)
        if self.use_mps:
            try:
                import paddle
                if paddle.device.is_compiled_with_mps():
                    paddle.device.set_device("mps")
                    return "mps"
            except Exception:
                pass

        return "cpu"

    def recognize(self, image_path: str) -> OCRResult:
        """
        识别单张图片中的文字

        Args:
            image_path: 图片路径

        Returns:
            OCRResult 包含识别文本和置信度
        """
        self._lazy_init()

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        try:
            result = self._ocr.ocr(image_path, cls=self.use_angle_cls)

            if not result or not result[0]:
                return OCRResult(text="", confidence=0.0)

            # 提取文字和置信度
            texts = []
            confidences = []
            boxes = []

            for line in result[0]:
                box, (text, conf) = line
                texts.append(text)
                confidences.append(conf)
                boxes.append([int(p[0]) for p in box])

            # 合并文本
            full_text = "\n".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                boxes=boxes
            )

        except Exception as e:
            logger.error(f"[OCREngine] 识别失败: {e}")
            return OCRResult(text="", confidence=0.0)

    def recognize_batch(self, image_paths: List[str]) -> List[OCRResult]:
        """
        批量识别图片

        Args:
            image_paths: 图片路径列表

        Returns:
            OCRResult 列表
        """
        return [self.recognize(path) for path in image_paths]

    def is_available(self) -> bool:
        """检查 OCR 引擎是否可用"""
        try:
            from paddleocr import PaddleOCR
            return True
        except ImportError:
            return False


def check_ocr_availability() -> dict:
    """
    检查 OCR 可用性

    Returns:
        dict 包含可用状态和相关信息
    """
    result = {
        "available": False,
        "engine": None,
        "device": "none",
        "error": None
    }

    try:
        from paddleocr import PaddleOCR
        result["engine"] = "PaddleOCR"
        result["available"] = True

        # 检测设备
        import paddle
        if paddle.device.is_compiled_with_cuda():
            result["device"] = "gpu"
        elif paddle.device.is_compiled_with_mps():
            result["device"] = "mps"
        else:
            result["device"] = "cpu"

    except ImportError as e:
        result["error"] = f"PaddleOCR 未安装: {e}"
    except Exception as e:
        result["error"] = str(e)

    return result


if __name__ == "__main__":
    # 测试代码
    import argparse

    parser = argparse.ArgumentParser(description="OCR 测试")
    parser.add_argument("image", help="图片路径")
    parser.add_argument("--check", action="store_true", help="检查可用性")
    args = parser.parse_args()

    if args.check:
        status = check_ocr_availability()
        print(f"OCR 可用性: {status}")
    else:
        engine = OCREngine()
        result = engine.recognize(args.image)
        print(f"识别结果:\n{result.text}")
        print(f"置信度: {result.confidence:.2f}")