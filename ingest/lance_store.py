"""
LanceDB 向量存储
- Serverless，查询时才占用内存
- 支持向量检索 + 元数据过滤
"""
import os
import sys
from pathlib import Path
from typing import Optional, Iterator, List
from dataclasses import dataclass, asdict
import numpy as np

import lancedb
import pyarrow as pa

# ==================== 常驻 Embedding 服务客户端 ====================

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
            print(f"[LanceStore] 常驻 Embedding 服务：编码 {len(texts)} 条文本", file=sys.stderr)
            return embeddings
        else:
            print(f"[LanceStore] Embedding 服务返回异常状态码：{resp.status_code}", file=sys.stderr)
            
    except Exception as e:
        print(f"[LanceStore] ⚠️ Embedding 常驻服务调用失败，降级到本地：{e}", file=sys.stderr)
    
    return None


@dataclass
class Document:
    """存储的文档结构"""
    id: str                    # 唯一 ID（source + chunk_index）
    text: str                  # 完整文本（含元数据前缀）
    raw_text: str              # 原始文本
    source: str                # 来源文件路径
    filename: str              # 文件名
    h1: str                    # 一级标题
    h2: str                    # 二级标题
    category: str              # 分类（来自 frontmatter）
    tags: str                  # 标签（逗号分隔）
    char_count: int            # 字符数
    vector: list[float]        # 向量


class LanceStore:
    """LanceDB 存储封装"""
    
    TABLE_NAME = "documents"
    
    def __init__(
        self,
        db_path: str = "~/zhiwei-rag/data/lance_db",
        embedding_manager=None
    ):
        self.db_path = os.path.expanduser(db_path)
        os.makedirs(self.db_path, exist_ok=True)
        
        self.db = lancedb.connect(self.db_path)
        self.embedding_manager = embedding_manager
        self._table = None
    
    @property
    def table(self):
        """获取或创建表"""
        if self._table is not None:
            return self._table
        
        if self.TABLE_NAME in self.db.table_names():
            self._table = self.db.open_table(self.TABLE_NAME)
        
        return self._table
    
    def create_table(self, dimension: int = 1024):
        """创建表（如果不存在）"""
        if self.TABLE_NAME in self.db.table_names():
            print(f"[LanceStore] 表已存在: {self.TABLE_NAME}")
            self._table = self.db.open_table(self.TABLE_NAME)
            return
        
        # 定义 schema
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("raw_text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("h1", pa.string()),
            pa.field("h2", pa.string()),
            pa.field("category", pa.string()),
            pa.field("tags", pa.string()),
            pa.field("char_count", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), dimension)),
        ])
        
        # 创建空表
        self._table = self.db.create_table(
            self.TABLE_NAME,
            schema=schema,
            mode="overwrite"
        )
        print(f"[LanceStore] 创建表: {self.TABLE_NAME}, 维度: {dimension}")
    
    def add_documents(self, docs: list[Document], batch_size: int = 100):
        """批量添加文档"""
        if not docs:
            return
        
        if self.table is None:
            dim = len(docs[0].vector)
            self.create_table(dimension=dim)
        
        # 转换为字典列表
        records = []
        for doc in docs:
            record = asdict(doc)
            records.append(record)
        
        # 分批写入
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.table.add(batch)
        
        print(f"[LanceStore] 写入 {len(docs)} 条文档")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_sql: Optional[str] = None
    ) -> list[dict]:
        """
        向量检索
        
        Args:
            query_vector: 查询向量
            top_k: 返回数量
            filter_sql: 过滤条件，如 "category = 'AI研报'"
            
        Returns:
            检索结果列表，每个元素包含 _distance 字段
        """
        if self.table is None:
            return []
        
        query = self.table.search(query_vector.tolist())
        
        if filter_sql:
            query = query.where(filter_sql)
        
        results = query.limit(top_k).to_list()
        
        return results
    
    def search_text(
        self,
        query_text: str,
        top_k: int = 10,
        filter_sql: Optional[str] = None
    ) -> list[dict]:
        """文本检索（优先常驻服务，降级本地）"""
        # 先尝试常驻 Embedding 服务
        embeddings = call_embed_service([query_text])
        
        if embeddings is not None and len(embeddings) > 0:
            query_vector = np.array(embeddings[0])
        else:
            # 降级到本地
            print(f"[LanceStore] 降级到本地 Embedding", file=sys.stderr)
            if self.embedding_manager is None:
                raise ValueError("需要 embedding_manager 才能进行文本检索")
            query_vector = self.embedding_manager.encode_single(query_text)
        
        return self.search(query_vector, top_k, filter_sql)
    
    def count(self) -> int:
        """文档数量"""
        if self.table is None:
            return 0
        return self.table.count_rows()
    
    def delete_by_source(self, source: str):
        """删除指定来源的文档（用于增量更新）"""
        if self.table is None:
            return
        self.table.delete(f"source = '{source}'")
    
    def clear(self):
        """清空所有数据"""
        if self.TABLE_NAME in self.db.table_names():
            self.db.drop_table(self.TABLE_NAME)
            self._table = None
            print("[LanceStore] 已清空表")
