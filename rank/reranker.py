"""
Cross-Encoder 精排器
- bge-reranker-base 模型（默认）
- MPS 加速
- 按需加载，用完释放
- 支持多种 Reranker 切换（Cohere / Voyage）⭐ v46.0 新增

使用：
    from rank.reranker import RerankerFactory

    # 创建 BGE Reranker（默认）
    reranker = RerankerFactory.create("bge")

    # 创建 Cohere Reranker
    reranker = RerankerFactory.create("cohere", api_key="xxx")

    # 精排
    results = reranker.rerank(query, search_results, top_k=5)
"""
import gc
import os
import sys
import time
from typing import Optional
from dataclasses import dataclass

import torch


@dataclass
class RerankResult:
    """精排结果"""
    text: str
    raw_text: str
    source: str
    original_score: float  # 原始分数（来自检索）
    rerank_score: float    # 精排分数
    track: str
    metadata: dict


class Reranker:
    """Cross-Encoder 精排器"""
    
    def __init__(
        self,
        model_name: str = 'BAAI/bge-reranker-base',
        device: str = 'mps',
        batch_size: int = 8,
        score_threshold: float = 0.01
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        
        self._model = None
        self._tokenizer = None
    
    def _rerank_core(
        self,
        query: str,
        results: list,  # list[RetrievalResult]
        top_k: int = 5
    ) -> list[RerankResult]:
        """
        精排核心逻辑（不含模型加载/卸载）
        
        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回数量
            
        Returns:
            精排后的结果列表
        """
        # 过滤掉 text 为 None 的结果
        valid_results = [r for r in results if r.text]
        
        if not valid_results:
            print(f"[Reranker] 没有有效的检索结果", file=sys.stderr)
            return []
        
        # 构建 query-passage 对
        pairs = [(query, r.text) for r in valid_results]
        
        # 分批计算分数
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            batch_scores = self._compute_scores(batch)
            scores.extend(batch_scores)
        
        # 组装结果（使用 valid_results 而不是原始 results）
        scored_results = []
        for r, score in zip(valid_results, scores):
            scored_results.append(RerankResult(
                text=r.text,
                raw_text=r.raw_text,
                source=r.source,
                original_score=r.score,
                rerank_score=score,
                track=r.track,
                metadata=r.metadata
            ))
        
        # 按精排分数排序
        scored_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # 过滤低分
        filtered = [
            r for r in scored_results
            if r.rerank_score >= self.score_threshold
        ]
        
        # 方案 B: 智能阈值保障 —— 如果过滤后结果太少，降级返回 Top-3，避免空结果
        if len(filtered) < 3 and len(scored_results) > 0:
            # 至少返回前 3 条（如果原始结果足够），或全部原始结果
            filtered = scored_results[:min(3, len(scored_results))]
        
        return filtered[:top_k]
    
    def rerank(
        self,
        query: str,
        results: list,  # list[RetrievalResult]
        top_k: int = 5
    ) -> list[RerankResult]:
        """
        对检索结果进行精排（标准模式：加载模型 -> 执行 -> 释放）
        
        Args:
            query: 查询文本
            results: 检索结果列表（需有 text, raw_text, source, score, track, metadata 属性）
            top_k: 返回数量
            
        Returns:
            精排后的结果列表
        """
        if not results:
            return []
        
        print(f"[Reranker] 输入 {len(results)} 条，精排中...", file=sys.stderr)
        start_time = time.time()
        
        try:
            # 加载模型
            self._load_model()
            
            # 执行精排
            result = self._rerank_core(query, results, top_k)
            
            elapsed = time.time() - start_time
            print(f"[Reranker] 完成，耗时 {elapsed:.2f}s，最终返回 {len(result)} 条", file=sys.stderr)
            
            return result
            
        finally:
            # 释放模型
            self._unload_model()
    
    def rerank_without_unload(
        self,
        query: str,
        results: list,  # list[RetrievalResult]
        top_k: int = 5
    ) -> list[RerankResult]:
        """
        对检索结果进行精排（常驻模式：不释放模型）
        
        供 FastAPI 服务使用，模型加载后保持常驻
        
        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回数量
            
        Returns:
            精排后的结果列表
        """
        if not results:
            return []
        
        print(f"[Reranker] 输入 {len(results)} 条，精排中 (常驻模式)...", file=sys.stderr)
        start_time = time.time()
        
        # 确保模型已加载
        self._load_model()
        
        # 执行精排（不释放模型）
        result = self._rerank_core(query, results, top_k)
        
        elapsed = time.time() - start_time
        print(f"[Reranker] 完成，耗时 {elapsed:.2f}s，最终返回 {len(result)} 条", file=sys.stderr)
        
        return result
    
    def _load_model(self):
        """加载模型"""
        if self._model is not None:
            return
        
        print(f"[Reranker] 加载模型: {self.model_name}", file=sys.stderr)
        start = time.time()
        
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        
        # 移动到指定设备
        if self.device == 'mps' and torch.backends.mps.is_available():
            self._model = self._model.to('mps')
        elif self.device == 'cuda' and torch.cuda.is_available():
            self._model = self._model.to('cuda')
        else:
            self._model = self._model.to('cpu')
        
        self._model.eval()
        
        elapsed = time.time() - start
        print(f"[Reranker] 模型加载完成，耗时 {elapsed:.1f}s", file=sys.stderr)
    
    def _compute_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """计算一批 query-passage 对的分数"""
        with torch.no_grad():
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # 移动到模型所在设备
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            outputs = self._model(**inputs)
            scores = outputs.logits.squeeze(-1)
            
            # 如果是单个样本，确保是列表
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            
            # Sigmoid 转换为 0-1 分数
            scores = torch.sigmoid(scores)
            
            return scores.cpu().tolist()
    
    def _unload_model(self):
        """释放模型"""
        if self._model is None:
            return
        
        print("[Reranker] 释放模型内存", file=sys.stderr)
        
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        
        gc.collect()
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


# 快捷函数
def rerank(query: str, results: list, top_k: int = 5, **kwargs) -> list[RerankResult]:
    """快捷精排函数"""
    reranker = Reranker(**kwargs)
    return reranker.rerank(query, results, top_k=top_k)


# ============================================================
# 多 Reranker 支持 ⭐ v46.0 新增
# ============================================================

class BaseReranker:
    """
    Reranker 抽象基类

    所有 Reranker 实现都需要继承此类。
    """

    def rerank(
        self,
        query: str,
        results: list,  # list[RetrievalResult]
        top_k: int = 5
    ) -> list[RerankResult]:
        """
        对检索结果进行精排

        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回数量

        Returns:
            精排后的结果列表
        """
        raise NotImplementedError

    def rerank_without_unload(
        self,
        query: str,
        results: list,
        top_k: int = 5
    ) -> list[RerankResult]:
        """
        常驻模式精排（不释放模型）

        供 FastAPI 服务使用。
        """
        return self.rerank(query, results, top_k)


class BgeReranker(BaseReranker):
    """
    BGE Reranker（本地模型）

    使用 BAAI/bge-reranker-base 进行精排。
    支持本地推理，无需 API 调用。
    """

    def __init__(
        self,
        model_name: str = 'BAAI/bge-reranker-base',
        device: str = 'mps',
        batch_size: int = 8,
        score_threshold: float = 0.01
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.score_threshold = score_threshold

        # 委托给原有 Reranker 实现
        self._impl = Reranker(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            score_threshold=score_threshold
        )

    def rerank(
        self,
        query: str,
        results: list,
        top_k: int = 5
    ) -> list[RerankResult]:
        return self._impl.rerank(query, results, top_k)

    def rerank_without_unload(
        self,
        query: str,
        results: list,
        top_k: int = 5
    ) -> list[RerankResult]:
        return self._impl.rerank_without_unload(query, results, top_k)


class CohereReranker(BaseReranker):
    """
    Cohere Reranker（云端 API）

    使用 Cohere API 进行精排。
    需要配置 COHERE_API_KEY 环境变量。

    优点：
    - 多语言支持好
    - 不占用本地资源
    - 质量较高

    缺点：
    - 需要 API 调用
    - 有调用成本
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = 'rerank-multilingual-v3.0',
        score_threshold: float = 0.01
    ):
        self.api_key = api_key or os.environ.get('COHERE_API_KEY')
        self.model = model
        self.score_threshold = score_threshold

        if not self.api_key:
            raise ValueError("Cohere API Key 未配置。请设置 COHERE_API_KEY 环境变量")

    def rerank(
        self,
        query: str,
        results: list,
        top_k: int = 5
    ) -> list[RerankResult]:
        if not results:
            return []

        try:
            import cohere

            client = cohere.Client(self.api_key)

            # 提取文本
            docs = [r.text for r in results if r.text]

            if not docs:
                return []

            # 调用 Cohere API
            response = client.rerank(
                query=query,
                documents=docs,
                top_n=min(top_k, len(docs)),
                model=self.model
            )

            # 构建结果
            reranked = []
            for item in response.results:
                idx = item.index
                if idx < len(results):
                    r = results[idx]
                    reranked.append(RerankResult(
                        text=r.text,
                        raw_text=r.raw_text,
                        source=r.source,
                        original_score=r.score,
                        rerank_score=item.relevance_score,
                        track=r.track,
                        metadata=r.metadata
                    ))

            return reranked[:top_k]

        except ImportError:
            print("[CohereReranker] cohere 包未安装，请运行: pip install cohere", file=sys.stderr)
            return []
        except Exception as e:
            print(f"[CohereReranker] API 调用失败: {e}", file=sys.stderr)
            return []


class VoyageReranker(BaseReranker):
    """
    Voyage Reranker（云端 API）

    使用 Voyage AI API 进行精排。
    需要配置 VOYAGE_API_KEY 环境变量。

    优点：
    - 高质量精排
    - 支持长文本

    缺点：
    - 需要 API 调用
    - 有调用成本
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = 'rerank-2',
        score_threshold: float = 0.01
    ):
        self.api_key = api_key or os.environ.get('VOYAGE_API_KEY')
        self.model = model
        self.score_threshold = score_threshold

        if not self.api_key:
            raise ValueError("Voyage API Key 未配置。请设置 VOYAGE_API_KEY 环境变量")

    def rerank(
        self,
        query: str,
        results: list,
        top_k: int = 5
    ) -> list[RerankResult]:
        if not results:
            return []

        try:
            import voyageai

            client = voyageai.Client(self.api_key)

            # 提取文本
            docs = [r.text for r in results if r.text]

            if not docs:
                return []

            # 调用 Voyage API
            response = client.rerank(
                query=query,
                documents=docs,
                model=self.model,
                top_k=min(top_k, len(docs))
            )

            # 构建结果
            reranked = []
            for item in response.results:
                idx = item.index
                if idx < len(results):
                    r = results[idx]
                    reranked.append(RerankResult(
                        text=r.text,
                        raw_text=r.raw_text,
                        source=r.source,
                        original_score=r.score,
                        rerank_score=item.relevance_score,
                        track=r.track,
                        metadata=r.metadata
                    ))

            return reranked[:top_k]

        except ImportError:
            print("[VoyageReranker] voyageai 包未安装，请运行: pip install voyageai", file=sys.stderr)
            return []
        except Exception as e:
            print(f"[VoyageReranker] API 调用失败: {e}", file=sys.stderr)
            return []


class RerankerFactory:
    """
    Reranker 工厂

    用于创建不同类型的 Reranker 实例。

    使用：
        # BGE（默认，本地）
        reranker = RerankerFactory.create("bge")

        # Cohere（云端）
        reranker = RerankerFactory.create("cohere", api_key="xxx")

        # Voyage（云端）
        reranker = RerankerFactory.create("voyage", api_key="xxx")
    """

    @staticmethod
    def create(
        reranker_type: str = "bge",
        **kwargs
    ) -> BaseReranker:
        """
        创建 Reranker 实例

        Args:
            reranker_type: Reranker 类型（bge / cohere / voyage）
            **kwargs: 传递给具体 Reranker 的参数

        Returns:
            Reranker 实例
        """
        reranker_type = reranker_type.lower()

        if reranker_type == "bge":
            return BgeReranker(**kwargs)
        elif reranker_type == "cohere":
            return CohereReranker(**kwargs)
        elif reranker_type == "voyage":
            return VoyageReranker(**kwargs)
        else:
            raise ValueError(f"未知的 Reranker 类型: {reranker_type}。支持: bge, cohere, voyage")

    @staticmethod
    def create_from_config(config: dict) -> BaseReranker:
        """
        从配置字典创建 Reranker

        Args:
            config: 配置字典，包含 type 和其他参数

        Returns:
            Reranker 实例
        """
        reranker_type = config.get("type", "bge")
        kwargs = {k: v for k, v in config.items() if k != "type"}
        return RerankerFactory.create(reranker_type, **kwargs)
