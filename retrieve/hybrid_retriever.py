"""
三轨混合检索器 + 精排
"""
import hashlib
import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from retrieve.vector_track import VectorTrack, RetrievalResult
from retrieve.fts_track import FTSTrack
from retrieve.graph_track import GraphTrack
from retrieve.embedding_manager import EmbeddingManager
from rank.reranker import Reranker, RerankResult

# ==================== 常驻服务客户端 ====================

_RERANK_SERVICE_URL = "http://127.0.0.1:8765/rerank"
_EMBED_SERVICE_URL = "http://127.0.0.1:8765/embed"


def call_embed_service(texts: list[str]) -> Optional[list[list[float]]]:
    """
    调用常驻 Embedding 服务
    
    Args:
        texts: 文本列表
        
    Returns:
        向量列表，失败返回 None
    """
    try:
        import requests
        
        resp = requests.post(
            _EMBED_SERVICE_URL,
            json={"texts": texts},
            timeout=60
        )
        
        if resp.status_code == 200:
            data = resp.json()
            embeddings = data.get("embeddings", [])
            print(f"[Hybrid] 常驻 Embedding 服务：编码 {len(texts)} 条文本", file=sys.stderr)
            return embeddings
        else:
            print(f"[Hybrid] Embedding 服务返回异常状态码：{resp.status_code}", file=sys.stderr)
            
    except Exception as e:
        print(f"[Hybrid] ⚠️ Embedding 常驻服务调用失败，降级到本地：{e}", file=sys.stderr)
    
    return None


def _call_rerank_service(
    query: str,
    results: List[RetrievalResult],
    top_k: int = 5,
    score_threshold: float = 0.01
) -> Optional[List[RerankResult]]:
    """
    调用常驻精排服务
    
    Args:
        query: 查询文本
        results: 检索结果列表
        top_k: 返回数量
        score_threshold: 分数阈值
        
    Returns:
        精排后的结果列表，失败返回 None
    """
    try:
        import requests
        
        # 过滤掉 text 为 None 的结果
        valid_results = [r for r in results if r.text]
        
        if not valid_results:
            print(f"[Hybrid] 没有有效的检索结果，跳过精排", file=sys.stderr)
            return None
        
        # 转换为字典列表
        results_dict = [
            {
                'text': r.text,
                'raw_text': r.raw_text or r.text,
                'source': r.source,
                'score': r.score,
                'track': r.track,
                'metadata': r.metadata
            }
            for r in valid_results
        ]
        
        resp = requests.post(
            _RERANK_SERVICE_URL,
            json={"query": query, "results": results_dict, "top_k": top_k},
            timeout=30
        )
        
        if resp.status_code == 200:
            data = resp.json()
            # 转换为 RerankResult 对象
            output = []
            for item in data:
                output.append(RerankResult(
                    text=item['text'],
                    raw_text=item['raw_text'],
                    source=item['source'],
                    original_score=item['original_score'],
                    rerank_score=item['rerank_score'],
                    track=item['track'],
                    metadata=item['metadata']
                ))
            print(f"[Hybrid] 常驻精排服务：返回 {len(output)} 条", file=sys.stderr)
            return output
        else:
            print(f"[Hybrid] 精排服务返回异常状态码：{resp.status_code}", file=sys.stderr)
            
    except Exception as e:
        print(f"[Hybrid] ⚠️ 常驻服务调用失败，降级到本地：{e}", file=sys.stderr)
    
    # 降级：本地精排
    return None


@dataclass
class HybridConfig:
    """混合检索配置"""
    # 各轨道召回数量
    vector_top_k: int = 15
    fts_top_k: int = 10
    graph_top_k: int = 5
    
    # 轨道权重
    vector_weight: float = 0.5
    fts_weight: float = 0.3
    graph_weight: float = 0.2
    
    # 轨道开关
    enable_vector: bool = True
    enable_fts: bool = True
    enable_graph: bool = True
    
    # 精排配置
    enable_rerank: bool = True
    rerank_top_k: int = 5
    rerank_threshold: float = 0.01


class HybridRetriever:
    """三轨混合检索器"""
    
    def __init__(
        self,
        config: Optional[HybridConfig] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        lance_db_path: str = "~/zhiwei-rag/data/lance_db",
        klib_db_path: str = "~/Documents/Library/klib.db",
        graph_db_path: str = "~/zhiwei-scheduler/graph_db"
    ):
        self.config = config or HybridConfig()
        self.embedding_manager = embedding_manager
        
        # 初始化各轨道
        if self.config.enable_vector:
            self.vector_track = VectorTrack(
                lance_db_path=lance_db_path,
                embedding_manager=embedding_manager
            )
        else:
            self.vector_track = None
        
        if self.config.enable_fts:
            self.fts_track = FTSTrack(db_path=klib_db_path)
        else:
            self.fts_track = None
        
        if self.config.enable_graph:
            self.graph_track = GraphTrack(graph_db_path=graph_db_path)
        else:
            self.graph_track = None
        
        # 精排器（按需创建）
        self.reranker = None
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_sql: Optional[str] = None,
        use_rerank: Optional[bool] = None
    ) -> list:
        """
        三轨混合检索 + 可选精排
        
        Args:
            query: 查询文本
            top_k: 最终返回数量
            filter_sql: 向量检索过滤条件
            use_rerank: 是否使用精排（默认使用配置）
            
        Returns:
            检索结果列表（RerankResult 或 RetrievalResult）
        """
        # 第一阶段：三轨召回
        all_results = self._multi_track_recall(query, filter_sql)
        
        # 去重
        deduplicated = self._deduplicate(all_results)
        print(f"[Hybrid] 去重后: {len(deduplicated)} 条", file=sys.stderr)
        
        # 判断是否精排
        should_rerank = use_rerank if use_rerank is not None else self.config.enable_rerank
        
        if should_rerank and len(deduplicated) > 0:
            # 第二阶段：精排
            return self._rerank(query, deduplicated, top_k)
        else:
            # 直接按原始分数排序返回
            deduplicated.sort(key=lambda x: x.score, reverse=True)
            return deduplicated[:top_k]
    
    def _multi_track_recall(
        self,
        query: str,
        filter_sql: Optional[str] = None
    ) -> list[RetrievalResult]:
        """三轨召回"""
        all_results = []
        
        # 轨道 A：向量检索
        if self.vector_track:
            try:
                results = self.vector_track.search(
                    query,
                    top_k=self.config.vector_top_k,
                    filter_sql=filter_sql
                )
                for r in results:
                    r.score *= self.config.vector_weight
                all_results.extend(results)
                print(f"[Hybrid] 向量轨道: {len(results)} 条", file=sys.stderr)
            except Exception as e:
                print(f"[Hybrid] 向量轨道错误: {e}", file=sys.stderr)
        
        # 轨道 B：FTS 检索
        if self.fts_track:
            try:
                results = self.fts_track.search(
                    query,
                    top_k=self.config.fts_top_k
                )
                for r in results:
                    r.score *= self.config.fts_weight
                all_results.extend(results)
                print(f"[Hybrid] FTS 轨道: {len(results)} 条", file=sys.stderr)
            except Exception as e:
                print(f"[Hybrid] FTS 轨道错误: {e}", file=sys.stderr)
        
        # 轨道 C：图谱检索
        if self.graph_track:
            try:
                results = self.graph_track.search(
                    query,
                    top_k=self.config.graph_top_k
                )
                for r in results:
                    r.score *= self.config.graph_weight
                all_results.extend(results)
                print(f"[Hybrid] 图谱轨道: {len(results)} 条", file=sys.stderr)
            except Exception as e:
                print(f"[Hybrid] 图谱轨道错误: {e}", file=sys.stderr)
        
        return all_results
    
    def _deduplicate(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """去重"""
        seen = {}
        
        for r in results:
            # 处理 raw_text 和 text 可能为 None 的情况
            text_to_hash = (r.raw_text or r.text or '')[:200]
            if not text_to_hash:
                # 如果都为空，使用 source 和 track 作为备用键
                text_to_hash = f"{r.source}:{r.track}"
            content_key = hashlib.md5(
                text_to_hash.encode()
            ).hexdigest()
            
            if content_key not in seen:
                seen[content_key] = r
            elif r.score > seen[content_key].score:
                seen[content_key] = r
        
        return list(seen.values())
    
    def _rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int
    ) -> list[RerankResult]:
        """精排（优先常驻服务，降级到本地）"""
        # 先尝试常驻服务
        service_result = _call_rerank_service(
            query,
            results,
            top_k=top_k,
            score_threshold=self.config.rerank_threshold
        )
        
        if service_result is not None:
            return service_result
        
        # 降级到本地精排
        print(f"[Hybrid] 降级到本地精排", file=sys.stderr)
        if self.reranker is None:
            self.reranker = Reranker(
                score_threshold=self.config.rerank_threshold
            )
        
        return self.reranker.rerank(
            query,
            results,
            top_k=top_k
        )
    
    def search_without_rerank(
        self,
        query: str,
        top_k: int = 10
    ) -> list[RetrievalResult]:
        """不带精排的检索"""
        return self.search(query, top_k=top_k, use_rerank=False)
    
    def search_vector_only(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """仅向量检索"""
        if not self.vector_track:
            return []
        return self.vector_track.search(query, top_k=top_k)
    
    def search_fts_only(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """仅 FTS 检索"""
        if not self.fts_track:
            return []
        return self.fts_track.search(query, top_k=top_k)
    
    def search_graph_only(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """仅图谱检索"""
        if not self.graph_track:
            return []
        return self.graph_track.search(query, top_k=top_k)


# 快捷函数
def hybrid_search(query: str, top_k: int = 5, **kwargs) -> list:
    """快捷混合检索（含精排）"""
    retriever = HybridRetriever(**kwargs)
    return retriever.search(query, top_k=top_k)
