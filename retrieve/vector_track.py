"""
轨道 A：LanceDB 向量检索 + 全文检索

支持两种检索模式：
- 向量检索：语义相似度搜索
- FTS 检索：关键词全文搜索（jieba 分词 + LanceDB FTS）
"""
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.lance_store import LanceStore
from retrieve.embedding_manager import EmbeddingManager


@dataclass
class RetrievalResult:
    """检索结果"""
    text: str              # 完整文本（含元数据前缀）
    raw_text: str          # 原始文本
    source: str            # 来源
    score: float           # 相关性分数（越高越好）
    track: str             # 来源轨道
    metadata: dict         # 额外元数据


class VectorTrack:
    """向量检索 + 全文检索轨道"""

    def __init__(
        self,
        lance_db_path: str = "~/zhiwei-rag/data/lance_db",
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.store = LanceStore(
            db_path=lance_db_path,
            embedding_manager=self.embedding_manager
        )

    def search(
        self,
        query: str,
        top_k: int = 15,
        filter_sql: Optional[str] = None
    ) -> list[RetrievalResult]:
        """
        向量检索

        Args:
            query: 查询文本
            top_k: 返回数量
            filter_sql: 可选过滤条件

        Returns:
            检索结果列表，按相关性排序
        """
        results = self.store.search_text(query, top_k=top_k, filter_sql=filter_sql)

        output = []
        for r in results:
            # LanceDB 返回的 _distance 是 L2 距离，转换为相似度分数
            # 距离越小越相似，转换为 0-1 分数
            distance = r.get('_distance', 0)
            score = 1 / (1 + distance)  # 简单转换

            output.append(RetrievalResult(
                text=r['text'],
                raw_text=r['raw_text'],
                source=r['source'],
                score=score,
                track='vector',
                metadata={
                    'filename': r.get('filename', ''),
                    'h1': r.get('h1', ''),
                    'h2': r.get('h2', ''),
                    'category': r.get('category', ''),
                    'tags': r.get('tags', ''),
                    # MM-005: 引用标注字段
                    'chunk_type': r.get('chunk_type', 'text'),
                    'page': r.get('page', 0),
                    'figure_path': r.get('figure_path', ''),
                }
            ))

        return output

    def search_fts(
        self,
        query: str,
        top_k: int = 10,
        filter_sql: Optional[str] = None
    ) -> list[RetrievalResult]:
        """
        全文检索（FTS）

        使用 jieba 分词 + LanceDB FTS 实现。
        适用于关键词精确匹配场景。

        Args:
            query: 查询文本
            top_k: 返回数量
            filter_sql: 可选过滤条件

        Returns:
            检索结果列表，按 FTS 分数排序
        """
        results = self.store.search_fts(query, top_k=top_k, filter_sql=filter_sql)

        output = []
        for r in results:
            # LanceDB FTS 返回 _score 字段
            # 归一化为 0-1 分数
            fts_score = r.get('_score', 0)
            # BM25 分数通常是正数，但范围不确定
            # 简单归一化：假设最高分约为 top_k
            score = min(1.0, fts_score / 10.0) if fts_score > 0 else 0.1

            output.append(RetrievalResult(
                text=r['text'],
                raw_text=r['raw_text'],
                source=r['source'],
                score=score,
                track='fts',
                metadata={
                    'filename': r.get('filename', ''),
                    'h1': r.get('h1', ''),
                    'h2': r.get('h2', ''),
                    'category': r.get('category', ''),
                    'tags': r.get('tags', ''),
                }
            ))

        return output
