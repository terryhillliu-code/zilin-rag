"""
三轨混合检索器 + 精排
"""
import hashlib
import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from retrieve.vector_track import VectorTrack, RetrievalResult
from retrieve.embedding_manager import EmbeddingManager
from rank.reranker import Reranker, RerankResult
from retrieve.query_rewriter import rewrite_query

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

    # RRF 融合参数 (v47.1 对齐 spec)
    rrf_k: int = 60  # RRF 公式参数

    # 轨道开关
    enable_vector: bool = True
    enable_fts: bool = True

    # 精排配置
    enable_rerank: bool = True
    rerank_top_k: int = 5
    rerank_threshold: float = 0.4  # v47.1: 从 0.01 对齐到 0.4

    # 图谱配置 (GraphTrack)
    enable_graph: bool = False
    graph_top_k: int = 5


class HybridRetriever:
    """三轨混合检索器

    轨道说明：
    - 向量轨道：LanceDB 向量检索（语义相似度）
    - FTS 轨道：LanceDB 全文检索（关键词匹配，jieba 分词）

    FTS-001 更新：FTS 轨道已从 klib.db 迁移到 LanceDB，
    实现真正的 Hybrid Search（同一数据源的向量 + FTS）。
    """

    def __init__(
        self,
        config: Optional[HybridConfig] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        lance_db_path: str = "~/zhiwei-rag/data/lance_db",
        klib_db_path: str = "~/Documents/Library/klib.db"
    ):
        self.config = config or HybridConfig()
        self.embedding_manager = embedding_manager
        self.lance_db_path = lance_db_path

        # 初始化各轨道
        # 轨道 A：向量检索（LanceDB）
        if self.config.enable_vector:
            self.vector_track = VectorTrack(
                lance_db_path=lance_db_path,
                embedding_manager=embedding_manager
            )
        else:
            self.vector_track = None

        # 轨道 B：FTS 检索（LanceDB，FTS-001 已迁移）
        # 使用同一个 VectorTrack 实例，调用其 FTS 方法
        if self.config.enable_fts:
            if self.vector_track:
                # 复用向量轨道实例
                self.lance_fts_track = self.vector_track
            else:
                # 如果向量轨道禁用，单独创建
                self.lance_fts_track = VectorTrack(
                    lance_db_path=lance_db_path,
                    embedding_manager=embedding_manager
                )
        else:
            self.lance_fts_track = None

        # 精排器（按需创建）
        self.reranker = None
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_sql: Optional[str] = None,
        use_rerank: Optional[bool] = None,
        use_rewrite: bool = True
    ) -> list:
        # 第一阶段：Query 重写 (v5.2)
        target_queries = [query]
        if use_rewrite:
            print(f"[Hybrid] 正在重写查询: {query}", file=sys.stderr)
            # 使用 broad 模式召回更多维度
            target_queries = rewrite_query(query, mode="broad")
            print(f"[Hybrid] 重写后查询: {target_queries}", file=sys.stderr)

        # 第二阶段：多轨道多查询召回
        all_recall_results = []
        for q in target_queries:
            results = self._multi_track_recall(q, filter_sql)
            all_recall_results.extend(results)

        # 第三阶段：RRF 融合
        fused = self._rrf_fusion(all_recall_results, k=self.config.rrf_k)
        print(f"[Hybrid] RRF 融合后: {len(fused)} 条", file=sys.stderr)

        # 去重（RRF 已处理同内容不同轨道的情况，这里做最终去重）
        deduplicated = self._deduplicate(fused)
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
                # v47.1: 移除权重乘法，改用 RRF 融合
                all_results.extend(results)
                print(f"[Hybrid] 向量轨道: {len(results)} 条", file=sys.stderr)
            except Exception as e:
                print(f"[Hybrid] 向量轨道错误: {e}", file=sys.stderr)

        # 轨道 B：FTS 检索（LanceDB）
        if self.lance_fts_track:
            try:
                results = self.lance_fts_track.search_fts(
                    query,
                    top_k=self.config.fts_top_k
                )
                # v47.1: 移除权重乘法，改用 RRF 融合
                all_results.extend(results)
                print(f"[Hybrid] FTS 轨道 (LanceDB): {len(results)} 条", file=sys.stderr)
            except Exception as e:
                print(f"[Hybrid] FTS 轨道错误: {e}", file=sys.stderr)
        
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

    def _rrf_fusion(
        self,
        results: list[RetrievalResult],
        k: int = 60
    ) -> list[RetrievalResult]:
        """
        RRF (Reciprocal Rank Fusion) 融合

        公式: Score = Σ 1/(k + rank)

        将各轨道结果按排名融合，消除分数量纲差异。

        Args:
            results: 各轨道检索结果（已标记 track 属性）
            k: RRF 参数，默认 60

        Returns:
            融合后的结果列表
        """
        # 按轨道分组
        track_results = {}
        for r in results:
            track = r.track or 'unknown'
            if track not in track_results:
                track_results[track] = []
            track_results[track].append(r)

        # 各轨道按分数排序
        for track in track_results:
            track_results[track].sort(key=lambda x: x.score, reverse=True)

        # RRF 融合
        fused_scores = {}
        result_map = {}

        for track, track_list in track_results.items():
            for rank, r in enumerate(track_list, 1):
                # 使用内容 hash 作为 key
                text_to_hash = (r.raw_text or r.text or '')[:200]
                if not text_to_hash:
                    text_to_hash = f"{r.source}:{r.track}"
                content_key = hashlib.md5(text_to_hash.encode()).hexdigest()

                if content_key not in fused_scores:
                    fused_scores[content_key] = 0.0
                    result_map[content_key] = r

                # RRF 公式
                fused_scores[content_key] += 1.0 / (k + rank)

        # 更新分数并排序
        for key, score in fused_scores.items():
            result_map[key].score = score

        return sorted(result_map.values(), key=lambda x: x.score, reverse=True)

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
        """仅 FTS 检索（LanceDB）"""
        if not self.lance_fts_track:
            return []
        return self.lance_fts_track.search_fts(query, top_k=top_k)

# 快捷函数
def hybrid_search(query: str, top_k: int = 5, **kwargs) -> list:
    """快捷混合检索（含精排）"""
    retriever = HybridRetriever(**kwargs)
    return retriever.search(query, top_k=top_k)
