"""
VLM 引擎 - 视觉语言模型封装

支持：
- 本地模型：Qwen2-VL, MiniCPM-V
- 云端降级：百炼 qwen-vl-plus
"""
import os
import sys
import base64
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# 云端 API 配置
BAILIAN_VL_API = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


@dataclass
class VLMResult:
    """VLM 结果"""
    description: str          # 图片描述
    model: str               # 使用的模型
    is_local: bool           # 是否本地模型
    tokens_used: int = 0     # 消耗的 token 数


class VLMEngine:
    """
    视觉语言模型引擎

    优先使用本地模型，失败时降级到云端 API
    """

    # 支持的本地模型
    LOCAL_MODELS = {
        "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
        "minicpm-v": "openbmb/MiniCPM-V-2_6",
    }

    # 云端模型
    CLOUD_MODELS = {
        "qwen-vl-plus": "qwen-vl-plus",
        "qwen-vl-max": "qwen-vl-max",
    }

    def __init__(
        self,
        model_name: str = "qwen2-vl-2b",
        prefer_local: bool = True,
        api_key: Optional[str] = None,
        device: str = "auto"
    ):
        """
        初始化 VLM 引擎

        Args:
            model_name: 模型名称
            prefer_local: 优先使用本地模型
            api_key: 云端 API 密钥（用于降级）
            device: 设备，'auto' / 'cuda' / 'mps' / 'cpu'
        """
        self.model_name = model_name
        self.prefer_local = prefer_local
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.device = device
        self._model = None
        self._processor = None
        self._initialized = False
        self._use_cloud = False

    def _lazy_init(self):
        """延迟初始化"""
        if self._initialized:
            return

        if self.prefer_local:
            success = self._init_local_model()
            if not success and self.api_key:
                logger.warning("[VLMEngine] 本地模型初始化失败，降级到云端 API")
                self._use_cloud = True

        self._initialized = True

    def _init_local_model(self) -> bool:
        """初始化本地模型"""
        try:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            from transformers import AutoProcessor, AutoModelForVision2Seq

            # 获取模型路径
            model_path = self.LOCAL_MODELS.get(self.model_name.lower())
            if not model_path:
                logger.warning(f"[VLMEngine] 未知模型: {self.model_name}")
                return False

            logger.info(f"[VLMEngine] 加载本地模型: {model_path}")

            # 检测设备
            device = self._detect_device()

            # 加载模型
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map=device if device != "cpu" else None
            )
            self._processor = AutoProcessor.from_pretrained(model_path)

            logger.info(f"[VLMEngine] 本地模型加载成功，设备: {device}")
            return True

        except ImportError as e:
            logger.warning(f"[VLMEngine] transformers 版本不支持: {e}")
            return False
        except Exception as e:
            logger.error(f"[VLMEngine] 本地模型加载失败: {e}")
            return False

    def _detect_device(self) -> str:
        """检测可用设备"""
        if self.device != "auto":
            return self.device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass

        return "cpu"

    def describe_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        max_tokens: int = 1024
    ) -> VLMResult:
        """
        生成图片描述

        Args:
            image_path: 图片路径
            prompt: 自定义提示词
            max_tokens: 最大输出 token 数

        Returns:
            VLMResult 包含描述信息
        """
        self._lazy_init()

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        default_prompt = "请详细描述这张图片的内容，包括文字、图表、物体等关键信息。"

        if self._use_cloud:
            return self._describe_with_cloud(image_path, prompt or default_prompt, max_tokens)
        else:
            return self._describe_with_local(image_path, prompt or default_prompt, max_tokens)

    def _describe_with_local(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int
    ) -> VLMResult:
        """使用本地模型生成描述"""
        try:
            from PIL import Image
            import torch

            # 加载图片
            image = Image.open(image_path).convert("RGB")

            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # 处理输入
            inputs = self._processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
                padding=True
            )

            # 移动到设备
            device = self._detect_device()
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # 生成
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens
                )

            # 解码
            description = self._processor.decode(outputs[0], skip_special_tokens=True)

            return VLMResult(
                description=description,
                model=self.model_name,
                is_local=True,
                tokens_used=len(outputs[0])
            )

        except Exception as e:
            logger.error(f"[VLMEngine] 本地推理失败: {e}")
            if self.api_key:
                logger.info("[VLMEngine] 降级到云端 API")
                return self._describe_with_cloud(image_path, prompt, max_tokens)
            raise

    def _describe_with_cloud(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int
    ) -> VLMResult:
        """使用云端 API 生成描述"""
        if not self.api_key:
            raise ValueError("需要 API 密钥才能使用云端服务")

        try:
            import requests
            from PIL import Image
            import io

            # 读取并编码图片
            with open(image_path, "rb") as f:
                image_data = f.read()

            # 转为 base64
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # 检测图片格式
            image_format = image_path.split(".")[-1].lower()
            if image_format == "jpg":
                image_format = "jpeg"

            # 构建请求
            payload = {
                "model": "qwen-vl-plus",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format};base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                BAILIAN_VL_API,
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                raise RuntimeError(f"API 调用失败: {response.status_code} - {response.text}")

            result = response.json()
            description = result["choices"][0]["message"]["content"]
            tokens_used = result.get("usage", {}).get("total_tokens", 0)

            return VLMResult(
                description=description,
                model="qwen-vl-plus",
                is_local=False,
                tokens_used=tokens_used
            )

        except Exception as e:
            logger.error(f"[VLMEngine] 云端 API 调用失败: {e}")
            raise

    def is_available(self) -> dict:
        """检查 VLM 可用性"""
        result = {
            "local_available": False,
            "cloud_available": bool(self.api_key),
            "device": self._detect_device() if self.prefer_local else "none"
        }

        if self.prefer_local:
            try:
                import torch
                import transformers
                result["local_available"] = True
                result["transformers_version"] = transformers.__version__
            except ImportError:
                pass

        return result


def check_vlm_availability() -> dict:
    """检查 VLM 系统可用性"""
    engine = VLMEngine()
    return engine.is_available()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLM 测试")
    parser.add_argument("image", help="图片路径")
    parser.add_argument("--prompt", default="请描述这张图片", help="提示词")
    parser.add_argument("--check", action="store_true", help="检查可用性")
    parser.add_argument("--cloud", action="store_true", help="强制使用云端 API")
    args = parser.parse_args()

    if args.check:
        status = check_vlm_availability()
        print(f"VLM 可用性: {json.dumps(status, indent=2, ensure_ascii=False)}")
    else:
        engine = VLMEngine(prefer_local=not args.cloud)
        result = engine.describe_image(args.image, args.prompt)
        print(f"模型: {result.model} ({'本地' if result.is_local else '云端'})")
        print(f"描述:\n{result.description}")