"""
轨道 B：klib FTS5 全文检索
复用现有的 klib.db
"""
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from retrieve.vector_track import RetrievalResult


class FTSTrack:
    """FTS5 全文检索轨道"""
    
    def __init__(self, db_path: str = "~/Documents/Library/klib.db"):
        self.db_path = os.path.expanduser(db_path)
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> list[RetrievalResult]:
        """
        FTS5 全文检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            检索结果列表
        """
        if not os.path.exists(self.db_path):
            print(f"[FTSTrack] 数据库不存在: {self.db_path}", file=sys.stderr)
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 检查 FTS 表是否存在
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='books_fts'
            """)
            if not cursor.fetchone():
                # 尝试从常规表查询（降级）
                print("[FTSTrack] 未找到 books_fts，使用 search_simple 降级", file=sys.stderr)
                conn.close()
                return self.search_simple(query, top_k)
            # FTS5 查询 (针对 klib.db 的实际 schema: title, author, toc, summary)
            cursor.execute("""
                SELECT 
                    title,
                    summary,
                    toc,
                    bm25(books_fts) as score
                FROM books_fts
                WHERE books_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """, (query, top_k))
            
            results = []
            for row in cursor.fetchall():
                title, summary, toc, score = row
                # 合并摘要和目录作为文本内容
                text_content = f"摘要: {summary[:300]}\n目录: {toc[:200]}"
                
                # BM25 分数是负数，越大（接近0）越相关
                normalized_score = 1 / (1 + abs(score))
                
                results.append(RetrievalResult(
                    text=f"{title}\n\n{text_content}",
                    raw_text=text_content,
                    source='klib.db',
                    score=normalized_score,
                    track='fts',
                    metadata={'title': title}
                ))
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"[FTSTrack] 查询错误: {e}", file=sys.stderr)
            return self.search_simple(query, top_k)
    
    def search_simple(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """
        简化版检索：直接 LIKE 匹配
        当 FTS5 不可用时的降级方案
        """
        if not os.path.exists(self.db_path):
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            results = []
            
            # 针对 klib.db 的 books 表进行 LIKE 查询
            cursor.execute("""
                SELECT title, summary, toc
                FROM books
                WHERE title LIKE ? OR summary LIKE ? OR toc LIKE ?
                LIMIT ?
            """, (f'%{query}%', f'%{query}%', f'%{query}%', top_k))
            
            for row in cursor.fetchall():
                title, summary, toc = row
                text_content = f"摘要: {summary[:300] if summary else ''}\n目录: {toc[:200] if toc else ''}"
                results.append(RetrievalResult(
                    text=f"{title}\n\n{text_content}",
                    raw_text=text_content,
                    source='klib.db',
                    score=0.5,
                    track='fts',
                    metadata={'title': title}
                ))
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"[FTSTrack] 简化查询错误: {e}", file=sys.stderr)
            return []
