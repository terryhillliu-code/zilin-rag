"""
Cross-Encoder 精排器
- bge-reranker-base 模型
- MPS 加速
- 按需加载，用完释放
"""
import gc
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
            results: 检索结果列表（需有 text, raw_text, source, score, track, metadata 属性）
            top_k: 返回数量
            
        Returns:
            精排后的结果列表
        """
        if not results:
            return []
        
        print(f"[Reranker] 输入 {len(results)} 条，精排中...", file=sys.stderr)
        start_time = time.time()
        
        # 加载模型
        self._load_model()
        
        try:
            # 构建 query-passage 对
            pairs = [(query, r.text) for r in results]
            
            # 分批计算分数
            scores = []
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                batch_scores = self._compute_scores(batch)
                scores.extend(batch_scores)
            
            # 组装结果
            scored_results = []
            for r, score in zip(results, scores):
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
            
            elapsed = time.time() - start_time
            print(f"[Reranker] 完成，耗时 {elapsed:.2f}s，最终返回 {len(filtered)} 条", file=sys.stderr)
            
            return filtered[:top_k]
            
        finally:
            # 释放模型
            self._unload_model()
    
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
