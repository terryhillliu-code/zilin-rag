"""
三轨混合检索器
合并向量、全文、图谱检索结果
"""
import hashlib
from typing import Optional
from dataclasses import dataclass, field

from retrieve.vector_track import VectorTrack, RetrievalResult
from retrieve.fts_track import FTSTrack
from retrieve.graph_track import GraphTrack
from retrieve.embedding_manager import EmbeddingManager


@dataclass
class HybridConfig:
    """混合检索配置"""
    vector_top_k: int = 15
    fts_top_k: int = 10
    graph_top_k: int = 5
    
    # 轨道权重（用于最终排序）
    vector_weight: float = 0.5
    fts_weight: float = 0.3
    graph_weight: float = 0.2
    
    # 是否启用各轨道
    enable_vector: bool = True
    enable_fts: bool = True
    enable_graph: bool = True


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
        
        # 初始化各轨道
        self.embedding_manager = embedding_manager
        
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
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_sql: Optional[str] = None
    ) -> list[RetrievalResult]:
        """
        三轨混合检索
        
        Args:
            query: 查询文本
            top_k: 最终返回数量
            filter_sql: 向量检索的过滤条件
            
        Returns:
            合并去重后的检索结果，按综合分数排序
        """
        all_results = []
        
        # 轨道 A：向量检索
        if self.vector_track:
            try:
                vector_results = self.vector_track.search(
                    query,
                    top_k=self.config.vector_top_k,
                    filter_sql=filter_sql
                )
                # 应用权重
                for r in vector_results:
                    r.score *= self.config.vector_weight
                all_results.extend(vector_results)
                print(f"[Hybrid] 向量轨道: {len(vector_results)} 条")
            except Exception as e:
                print(f"[Hybrid] 向量轨道错误: {e}")
        
        # 轨道 B：FTS 全文检索
        if self.fts_track:
            try:
                fts_results = self.fts_track.search(
                    query,
                    top_k=self.config.fts_top_k
                )
                for r in fts_results:
                    r.score *= self.config.fts_weight
                all_results.extend(fts_results)
                print(f"[Hybrid] FTS 轨道: {len(fts_results)} 条")
            except Exception as e:
                print(f"[Hybrid] FTS 轨道错误: {e}")
        
        # 轨道 C：图谱检索
        if self.graph_track:
            try:
                graph_results = self.graph_track.search(
                    query,
                    top_k=self.config.graph_top_k
                )
                for r in graph_results:
                    r.score *= self.config.graph_weight
                all_results.extend(graph_results)
                print(f"[Hybrid] 图谱轨道: {len(graph_results)} 条")
            except Exception as e:
                print(f"[Hybrid] 图谱轨道错误: {e}")
        
        # 去重
        deduplicated = self._deduplicate(all_results)
        print(f"[Hybrid] 去重后: {len(deduplicated)} 条")
        
        # 按分数排序
        deduplicated.sort(key=lambda x: x.score, reverse=True)
        
        return deduplicated[:top_k]
    
    def _deduplicate(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """
        去重：基于文本内容的哈希
        同一内容多次命中时，保留分数最高的
        """
        seen = {}
        
        for r in results:
            # 使用 raw_text 的前 200 字符做哈希
            content_key = hashlib.md5(
                r.raw_text[:200].encode()
            ).hexdigest()
            
            if content_key not in seen:
                seen[content_key] = r
            else:
                # 保留分数更高的
                if r.score > seen[content_key].score:
                    seen[content_key] = r
        
        return list(seen.values())
