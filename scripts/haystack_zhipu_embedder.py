#!/usr/bin/env python3
"""Haystack 兼容的智谱 Embedder 组件

使用：
    from scripts.haystack_zhipu_embedder import ZhipuTextEmbedder
    embedder = ZhipuTextEmbedder()
    result = embedder.run(text="MoE架构")
"""
from haystack.components.embedders import TextEmbedder
from haystack.dataclasses import EmbeddingResult
from scripts.zhipu_embedder import ZhipuEmbedder


class ZhipuTextEmbedder(TextEmbedder):
    """Haystack TextEmbedder 接口的智谱实现"""

    def __init__(self, api_key: str | None = None, batch_size: int = 10):
        self.embedder = ZhipuEmbedder(api_key=api_key, batch_size=batch_size)

    def run(self, text: str) -> dict:
        """生成单个文本的向量"""
        embedding = self.embedder.embed_single(text)
        return {"embedding": EmbeddingResult(embedding=embedding)}