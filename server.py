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


class HealthOutput(BaseModel):
    """健康检查响应"""
    status: str
    reranker_loaded: bool
    embedding_loaded: bool


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


@app.post("/rerank", response_model=list[RerankOutput])
async def rerank_endpoint(input: RerankInput):
    """
    精排服务
    
    接收检索结果，返回精排后的结果
    """
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


# ==================== 启动事件 ====================

@app.on_event("startup")
async def startup_event():
    """服务启动时预加载模型"""
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