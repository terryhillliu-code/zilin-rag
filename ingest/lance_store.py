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
import fcntl
import jieba
from contextlib import contextmanager

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


def tokenize_text(text: str) -> str:
    """
    jieba 分词（搜索引擎模式），用于 FTS 索引

    Args:
        text: 原始文本

    Returns:
        空格分隔的分词结果
    """
    if not text:
        return ""
    return " ".join(jieba.cut_for_search(text))


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
    tokenized_text: str = ""   # jieba 分词后文本（FTS 索引用）
    vector: list = None        # 向量
    # 多模态字段 (MM-001)
    chunk_type: str = "text"   # chunk 类型: "text" / "figure" / "table" / "frame"
    page: int = 0              # 页码（PDF/PPT）
    timestamp: str = ""        # 时间戳（视频），如 "03:25"
    figure_path: str = ""      # 图片文件路径（可选）

    def __post_init__(self):
        if self.vector is None:
            self.vector = []


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
        self.lock_file = os.path.join(self.db_path, "write.lock")

    @contextmanager
    def _write_lock(self):
        """文件锁上下文管理器，确保写入串行化"""
        lock_f = open(self.lock_file, "w")
        try:
            # 阻塞式获取排他锁
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            yield
        finally:
            # 释放锁并关闭文件
            fcntl.flock(lock_f, fcntl.LOCK_UN)
            lock_f.close()
    
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
            pa.field("tokenized_text", pa.string()),  # jieba 分词文本（FTS）
            pa.field("vector", pa.list_(pa.float32(), dimension)),
            # 多模态字段 (MM-001)
            pa.field("chunk_type", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("timestamp", pa.string()),
            pa.field("figure_path", pa.string()),
        ])
        
        # 创建空表
        with self._write_lock():
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
            dim = len(docs[0].vector) if docs[0].vector else 1024
            self.create_table(dimension=dim)

        # 转换为字典列表，自动添加 tokenized_text
        records = []
        for doc in docs:
            record = asdict(doc)
            # 如果 tokenized_text 为空，自动分词
            if not record.get('tokenized_text'):
                record['tokenized_text'] = tokenize_text(doc.raw_text or doc.text)
            records.append(record)

        # 分批写入
        with self._write_lock():
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
        """文本检索（优先本地 embedding_manager，避免 HTTP 死锁）"""
        # ⭐ v69.0: 当 embedding_manager 可用时，直接使用，避免 HTTP 调用导致死锁
        if self.embedding_manager is not None:
            query_vector = self.embedding_manager.encode_single(query_text)
            return self.search(query_vector, top_k, filter_sql)

        # 降级：尝试常驻 Embedding 服务
        embeddings = call_embed_service([query_text])

        if embeddings is not None and len(embeddings) > 0:
            query_vector = np.array(embeddings[0])
        else:
            raise ValueError("需要 embedding_manager 才能进行文本检索")

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
        with self._write_lock():
            self.table.delete(f"source = '{source}'")
    
    def clear(self):
        """清空所有数据"""
        if self.TABLE_NAME in self.db.table_names():
            with self._write_lock():
                self.db.drop_table(self.TABLE_NAME)
                self._table = None
            print("[LanceStore] 已清空表")

    def create_fts_index(self, field: str = "tokenized_text"):
        """
        创建全文检索索引

        Args:
            field: FTS 索引字段（默认 tokenized_text）
        """
        if self.table is None:
            print("[LanceStore] 表不存在，无法创建 FTS 索引", file=sys.stderr)
            return False

        try:
            self.table.create_fts_index(field)
            print(f"[LanceStore] FTS 索引创建成功: {field}")
            return True
        except Exception as e:
            # 如果索引已存在，LanceDB 会抛出异常
            if "already exists" in str(e).lower():
                print(f"[LanceStore] FTS 索引已存在: {field}")
                return True
            print(f"[LanceStore] FTS 索引创建失败: {e}", file=sys.stderr)
            return False

    def search_fts(
        self,
        query_text: str,
        top_k: int = 10,
        filter_sql: Optional[str] = None
    ) -> list[dict]:
        """
        全文检索（FTS）

        Args:
            query_text: 查询文本
            top_k: 返回数量
            filter_sql: 过滤条件

        Returns:
            检索结果列表
        """
        if self.table is None:
            return []

        # 对查询文本进行 jieba 分词
        tokenized_query = tokenize_text(query_text)

        try:
            search_query = self.table.search(tokenized_query, query_type="fts")
            if filter_sql:
                search_query = search_query.where(filter_sql)
            results = search_query.limit(top_k).to_list()
            return results
        except Exception as e:
            print(f"[LanceStore] FTS 搜索失败: {e}", file=sys.stderr)
            return []
