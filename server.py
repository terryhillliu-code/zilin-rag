#!/usr/bin/env python3
"""
zhiwei-rag FastAPI 常驻服务
提供 Embedding 和 Reranker 常驻推理能力

启动命令:
    ~/zhiwei-rag/venv/bin/python server.py

API:
    GET  /health    - 健康检查
    POST /rerank    - 精排服务
    POST /embed     - Embedding 服务
"""
import sys
import os
from pathlib import Path
from typing import Optional

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent))

# 加载全局密钥（必须在其他导入之前）
sys.path.insert(0, str(Path.home() / "scripts"))
from load_secrets import load_secrets
load_secrets(silent=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from rank.reranker import Reranker, RerankResult
from retrieve.vector_track import RetrievalResult
from retrieve.embedding_manager import EmbeddingManager

# ==================== 配置 ====================

HOST = "127.0.0.1"
PORT = 8765

# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="zhiwei-rag API",
    description="Embedding 和 Reranker 常驻推理服务",
    version="1.0.0"
)

# ==================== 全局常驻模型 ====================

_reranker: Optional[Reranker] = None
_embedding_manager: Optional[EmbeddingManager] = None


def get_reranker() -> Reranker:
    """获取全局常驻 Reranker 实例"""
    global _reranker
    if _reranker is None:
        print("[Server] 初始化 Reranker 模型...", file=sys.stderr)
        _reranker = Reranker(
            model_name='BAAI/bge-reranker-base',
            device='mps',
            batch_size=8,
            score_threshold=0.01
        )
        # 预加载模型
        _reranker._load_model()
        print("[Server] Reranker 模型预加载完成", file=sys.stderr)
    return _reranker


def get_embedding_manager() -> EmbeddingManager:
    """获取全局常驻 EmbeddingManager 实例"""
    global _embedding_manager
    if _embedding_manager is None:
        print("[Server] 初始化 EmbeddingManager...", file=sys.stderr)
        _embedding_manager = EmbeddingManager(
            model_name='BAAI/bge-large-zh-v1.5',
            device='mps',
            idle_timeout=0  # 不自动卸载，常驻
        )
        # 预加载模型
        _embedding_manager.preload()
        print("[Server] EmbeddingManager 预加载完成", file=sys.stderr)
    return _embedding_manager


# ==================== 数据模型 ====================

class RerankInput(BaseModel):
    """精排请求"""
    query: str
    results: list[dict]
    top_k: int = 5


class RerankOutput(BaseModel):
    """精排响应"""
    text: str
    raw_text: str
    source: str
    original_score: float
    rerank_score: float
    track: str
    metadata: dict


class EmbedInput(BaseModel):
    """Embedding 请求"""
    texts: list[str]


class EmbedOutput(BaseModel):
    """Embedding 响应"""
    embeddings: list[list[float]]
    dimension: int


class StatsOutput(BaseModel):
    """统计信息响应"""
    requests_total: int
    requests_rerank: int
    requests_embed: int
    reranker_loaded: bool
    embedding_loaded: bool
    uptime_seconds: float


class HealthOutput(BaseModel):
    """健康检查响应"""
    status: str
    reranker_loaded: bool
    embedding_loaded: bool


# ==================== 统计计数 ====================

_request_stats = {
    "total": 0,
    "rerank": 0,
    "embed": 0,
    "start_time": None
}


# ==================== API 接口 ====================

@app.get("/health", response_model=HealthOutput)
async def health_check():
    """健康检查"""
    reranker_loaded = _reranker is not None and _reranker._model is not None
    embedding_loaded = _embedding_manager is not None and _embedding_manager.is_loaded

    return HealthOutput(
        status="healthy",
        reranker_loaded=reranker_loaded,
        embedding_loaded=embedding_loaded
    )


@app.get("/stats", response_model=StatsOutput)
async def stats_endpoint():
    """
    统计信息

    返回服务运行统计数据
    """
    from time import time

    reranker_loaded = _reranker is not None and _reranker._model is not None
    embedding_loaded = _embedding_manager is not None and _embedding_manager.is_loaded

    uptime = 0.0
    if _request_stats["start_time"]:
        uptime = time() - _request_stats["start_time"]

    return StatsOutput(
        requests_total=_request_stats["total"],
        requests_rerank=_request_stats["rerank"],
        requests_embed=_request_stats["embed"],
        reranker_loaded=reranker_loaded,
        embedding_loaded=embedding_loaded,
        uptime_seconds=uptime
    )


@app.post("/rerank", response_model=list[RerankOutput])
async def rerank_endpoint(input: RerankInput):
    """
    精排服务

    接收检索结果，返回精排后的结果
    """
    _request_stats["total"] += 1
    _request_stats["rerank"] += 1

    try:
        reranker = get_reranker()
        
        # 转换为 RetrievalResult 对象
        retrieval_results = []
        for r in input.results:
            retrieval_results.append(RetrievalResult(
                text=r.get('text', ''),
                raw_text=r.get('raw_text', ''),
                source=r.get('source', ''),
                score=r.get('score', 0.0),
                track=r.get('track', 'unknown'),
                metadata=r.get('metadata', {})
            ))
        
        # 执行精排
        ranked_results = reranker.rerank_without_unload(
            query=input.query,
            results=retrieval_results,
            top_k=input.top_k
        )
        
        # 转换为 Pydantic 模型
        return [
            RerankOutput(
                text=r.text,
                raw_text=r.raw_text,
                source=r.source,
                original_score=r.original_score,
                rerank_score=r.rerank_score,
                track=r.track,
                metadata=r.metadata
            )
            for r in ranked_results
        ]
        
    except Exception as e:
        print(f"[Server] /rerank 错误：{e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed", response_model=EmbedOutput)
async def embed_endpoint(input: EmbedInput):
    """
    Embedding 服务

    接收文本列表，返回向量
    """
    _request_stats["total"] += 1
    _request_stats["embed"] += 1

    try:
        manager = get_embedding_manager()
        
        # 编码
        embeddings = manager.encode(input.texts)
        
        return EmbedOutput(
            embeddings=embeddings.tolist(),
            dimension=manager.dimension
        )
        
    except Exception as e:
        print(f"[Server] /embed 错误：{e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 检索端点 ====================

class SearchInput(BaseModel):
    query: str
    top_k: int = 5
    source_filter: str = ""

class SearchResult(BaseModel):
    content: str
    source: str
    score: float

class SearchOutput(BaseModel):
    results: list[SearchResult]

# 全局检索器（避免重复初始化）
_search_retriever = None

def get_search_retriever():
    """获取检索器（直接使用本地组件，避免 HTTP 死锁）"""
    global _search_retriever
    if _search_retriever is None:
        from retrieve.hybrid_retriever import HybridRetriever, HybridConfig
        config = HybridConfig(
            vector_top_k=15,
            fts_top_k=10,
            enable_rerank=True,
            rerank_top_k=5
        )
        # 传入本地的 EmbeddingManager，避免 HTTP 调用
        _search_retriever = HybridRetriever(
            config=config,
            embedding_manager=get_embedding_manager()
        )
    return _search_retriever

@app.post("/search", response_model=SearchOutput)
async def search_endpoint(input: SearchInput):
    """
    检索服务

    从知识库中检索相关内容
    """
    _request_stats["total"] += 1
    _request_stats["search"] = _request_stats.get("search", 0) + 1

    try:
        retriever = get_search_retriever()
        results = retriever.search(input.query, top_k=input.top_k, use_rerank=False)

        return SearchOutput(
            results=[
                SearchResult(
                    content=r.text if hasattr(r, 'text') else str(r),
                    source=r.source if hasattr(r, 'source') else "",
                    score=r.score if hasattr(r, 'score') else 0.0
                )
                for r in results
            ]
        )

    except Exception as e:
        print(f"[Server] /search 错误：{e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 启动事件 ====================

@app.on_event("startup")
async def startup_event():
    """服务启动时预加载模型"""
    from time import time
    _request_stats["start_time"] = time()

    print("[Server] 服务启动中...", file=sys.stderr)
    
    # 预加载 Reranker
    get_reranker()
    
    # 预加载 EmbeddingManager
    get_embedding_manager()
    
    print("[Server] 服务启动完成", file=sys.stderr)


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 确保日志目录存在
    log_dir = Path.home() / "logs"
    log_dir.mkdir(exist_ok=True)
    
    print(f"[Server] 启动服务：http://{HOST}:{PORT}", file=sys.stderr)
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )