#!/usr/bin/env python3
"""智谱 Embedder - 用于 Haystack Pipeline 混合检索

调用智谱 embedding-2 API，1024维向量，与 LanceDB 数据兼容

使用：
    from scripts.zhipu_embedder import ZhipuEmbedder
    embedder = ZhipuEmbedder()
    embeddings = embedder.embed(["文本1", "文本2"])
"""
import os
import requests

ZHIPU_API_KEY = os.environ.get('ZHIPU_API_KEY', '')
ZHIPU_API_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"


class ZhipuEmbedder:
    """智谱 Embedding API 客户端"""

    def __init__(self, api_key: str | None = None, batch_size: int = 10):
        self.api_key = api_key or ZHIPU_API_KEY
        self.batch_size = batch_size

        if not self.api_key:
            raise ValueError("请设置 ZHIPU_API_KEY 环境变量")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """生成文本向量"""
        if not texts:
            return []

        # 分批处理
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """单批 API 调用"""
        response = requests.post(
            ZHIPU_API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "embedding-2",
                "input": texts
            },
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"API错误 {response.status_code}: {response.text[:200]}")

        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_single(self, text: str) -> list[float]:
        """单个文本向量"""
        return self.embed([text])[0]


if __name__ == "__main__":
    # 测试
    embedder = ZhipuEmbedder()
    result = embedder.embed(["测试文本", "MoE架构"])
    print(f"✅ 生成 {len(result)} 个向量，维度 {len(result[0])}")