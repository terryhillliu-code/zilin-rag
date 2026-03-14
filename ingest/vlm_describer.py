"""
VLM 图片描述服务
- 云端百炼 qwen-vl-max
- 智能过滤无效图片
- 针对研报/论文场景优化 prompt
"""
import asyncio
import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from dotenv import load_dotenv
from openai import AsyncOpenAI

# 加载环境变量（zhiwei-bot 的 .env）
_env_path = Path.home() / "zhiwei-bot" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

if TYPE_CHECKING:
    from .pdf_image_extractor import ExtractedImage


@dataclass
class ImageDescription:
    """图片描述结果"""
    page: int                    # 页码（0-indexed）
    description: str             # 文字描述
    image_type: str              # "chart" | "table" | "diagram" | "photo" | "other"
    confidence: float            # 置信度 0-1
    skipped: bool = False        # 是否被过滤
    skip_reason: str = ""        # 跳过原因


VLM_PROMPT = """你是一个专业的研报/论文分析助手。请描述这张图片的内容。

要求：
1. 首先判断图片类型：chart（图表）、table（表格）、diagram（流程图/架构图）、photo（照片）、other（其他）
2. 如果是图表，提取关键数据和趋势
3. 如果是流程图/架构图，描述结构和关系
4. 如果是表格，提取主要数据（保留数字和百分比）
5. 使用简洁的中文描述，100-200 字

输出格式：
类型: {chart/table/diagram/photo/other}
描述: {详细描述}"""


class VLMDescriber:
    """VLM 图片描述服务"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-vl-max",
        timeout: int = 30,
        max_concurrent: int = 3
    ):
        """
        初始化 VLM 描述器

        Args:
            api_key: 百炼 API Key，默认从环境变量读取
            model: 模型名称，默认 qwen-vl-max
            timeout: 单张图片超时时间（秒）
            max_concurrent: 最大并发数，避免限流
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未配置 DASHSCOPE_API_KEY")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        self.timeout = timeout
        self.max_concurrent = max_concurrent

        # 过滤参数
        self.min_area = 10000        # 最小面积（100x100 像素）
        self.min_aspect_ratio = 0.1  # 最小宽高比
        self.max_aspect_ratio = 10   # 最大宽高比

    def should_process(self, image: 'ExtractedImage') -> tuple[bool, str]:
        """
        判断图片是否需要处理（过滤无效图片）

        Args:
            image: 提取的图片对象

        Returns:
            (是否处理, 原因说明)
        """
        # 面积太小（Logo、图标）
        area = image.width * image.height
        if area < self.min_area:
            return False, f"面积过小: {area} < {self.min_area}"

        # 宽高比极端（装饰线、边框）
        ratio = image.width / image.height
        if ratio < self.min_aspect_ratio:
            return False, f"宽高比过小: {ratio:.2f} < {self.min_aspect_ratio}"
        if ratio > self.max_aspect_ratio:
            return False, f"宽高比过大: {ratio:.2f} > {self.max_aspect_ratio}"

        return True, "有效图片"

    def _parse_response(self, text: str) -> tuple[str, str, float]:
        """
        解析 VLM 响应

        Args:
            text: VLM 返回的文本

        Returns:
            (image_type, description, confidence)
        """
        # 默认值
        image_type = "other"
        description = text
        confidence = 0.7

        # 尝试解析类型
        type_match = re.search(r"类型[：:]\s*(\w+)", text)
        if type_match:
            parsed_type = type_match.group(1).lower()
            if parsed_type in ["chart", "table", "diagram", "photo", "other"]:
                image_type = parsed_type
                confidence = 0.9

        # 尝试解析描述
        desc_match = re.search(r"描述[：:]\s*(.+)", text, re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()

        return image_type, description, confidence

    async def describe(
        self,
        image_bytes: bytes,
        page: int = 0,
        context: str = ""
    ) -> ImageDescription:
        """
        描述单张图片

        Args:
            image_bytes: 图片二进制数据
            page: 页码
            context: 上下文信息（如文件名、章节标题）

        Returns:
            ImageDescription 对象
        """
        try:
            # 压缩图片（如果太大）
            image_bytes = self._compress_image(image_bytes)

            # Base64 编码
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # 构建消息
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": VLM_PROMPT
                }
            ]

            # 调用 API
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=500
                ),
                timeout=self.timeout
            )

            # 解析响应
            text = response.choices[0].message.content
            image_type, description, confidence = self._parse_response(text)

            return ImageDescription(
                page=page,
                description=description,
                image_type=image_type,
                confidence=confidence
            )

        except asyncio.TimeoutError:
            return ImageDescription(
                page=page,
                description="",
                image_type="other",
                confidence=0.0,
                skipped=True,
                skip_reason="API 超时"
            )
        except Exception as e:
            return ImageDescription(
                page=page,
                description="",
                image_type="other",
                confidence=0.0,
                skipped=True,
                skip_reason=f"API 错误: {str(e)[:50]}"
            )

    def _compress_image(self, image_bytes: bytes, max_size: int = 512000) -> bytes:
        """
        压缩图片到指定大小以下

        Args:
            image_bytes: 原始图片数据
            max_size: 最大文件大小（字节），默认 500KB

        Returns:
            压缩后的图片数据
        """
        if len(image_bytes) <= max_size:
            return image_bytes

        try:
            from io import BytesIO
            from PIL import Image

            # 打开图片
            img = Image.open(BytesIO(image_bytes))

            # 转换为 RGB（如果是 RGBA）
            if img.mode == "RGBA":
                img = img.convert("RGB")

            # 计算缩放比例
            scale = (max_size / len(image_bytes)) ** 0.5
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)

            # 缩放
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 保存
            output = BytesIO()
            img.save(output, format="JPEG", quality=85)

            return output.getvalue()

        except Exception as e:
            print(f"[VLM] 图片压缩失败: {e}")
            return image_bytes

    async def describe_batch(
        self,
        images: List['ExtractedImage'],
        source: str = "",
        show_progress: bool = True
    ) -> List[ImageDescription]:
        """
        批量描述图片，带过滤和并发控制

        Args:
            images: 图片列表
            source: 来源文件名（用于日志）
            show_progress: 是否显示进度

        Returns:
            描述结果列表
        """
        results: List[ImageDescription] = []

        # 第一阶段：过滤
        valid_images = []
        for img in images:
            should, reason = self.should_process(img)
            if should:
                valid_images.append(img)
            else:
                results.append(ImageDescription(
                    page=img.page,
                    description="",
                    image_type="other",
                    confidence=0.0,
                    skipped=True,
                    skip_reason=reason
                ))

        if show_progress:
            print(f"[VLM] 有效图片: {len(valid_images)}/{len(images)}")

        # 第二阶段：并发处理（带信号量控制）
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_one(img: 'ExtractedImage') -> ImageDescription:
            async with semaphore:
                context = f"{source} 第{img.page + 1}页"
                return await self.describe(img.image_bytes, img.page, context)

        # 执行并发
        valid_results = await asyncio.gather(*[
            process_one(img) for img in valid_images
        ])

        # 合并结果（保持原顺序）
        valid_idx = 0
        result_idx = 0
        for img in images:
            should, _ = self.should_process(img)
            if should:
                results.insert(result_idx, valid_results[valid_idx])
                valid_idx += 1
            result_idx += 1

        # 重新排序（按页码）
        results.sort(key=lambda x: x.page)

        if show_progress:
            success = sum(1 for r in results if not r.skipped)
            skipped = sum(1 for r in results if r.skipped)
            print(f"[VLM] 完成: 成功 {success}, 跳过 {skipped}")

        return results


# ==================== 测试入口 ====================

async def _test():
    """测试 VLM 描述服务"""
    from .pdf_image_extractor import extract_images

    # 查找测试 PDF
    import glob
    search_paths = [
        "/Users/liufang/Documents/Library/reports/**/*.pdf",
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

    # 提取图片
    images = extract_images(test_pdf)
    print(f"提取图片: {len(images)} 张")

    if not images:
        print("无图片可处理")
        return

    # 创建描述器
    describer = VLMDescriber()

    # 测试过滤
    valid = [img for img in images if describer.should_process(img)[0]]
    print(f"有效图片: {len(valid)} 张")

    # 测试单张描述
    if valid:
        print(f"\n描述第 {valid[0].page + 1} 页图片...")
        desc = await describer.describe(
            valid[0].image_bytes,
            valid[0].page,
            f"研报第{valid[0].page + 1}页"
        )
        print(f"类型: {desc.image_type}")
        print(f"置信度: {desc.confidence}")
        print(f"描述: {desc.description[:200]}...")
        if desc.skipped:
            print(f"跳过原因: {desc.skip_reason}")


if __name__ == "__main__":
    asyncio.run(_test())