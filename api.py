"""
zhiwei-rag 统一 API
供其他模块调用
"""
import sys
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

from retrieve.hybrid_retriever import HybridRetriever, HybridConfig
from retrieve.embedding_manager import EmbeddingManager, get_embedding_manager
from generate.context_builder import ContextBuilder, ContextConfig
from rank.reranker import RerankResult
from retrieve.vector_track import RetrievalResult


@dataclass
class RAGConfig:
    """RAG 配置"""
    # 检索配置
    vector_top_k: int = 15
    fts_top_k: int = 10
    graph_top_k: int = 5
    
    # 精排配置
    enable_rerank: bool = True
    rerank_top_k: int = 5
    
    # 上下文配置
    max_context_tokens: int = 4000
    include_source: bool = True


class RAG:
    """RAG 统一接口"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        
        # 延迟初始化
        self._embedding_manager = None
        self._retriever = None
        self._context_builder = None
    
    @property
    def embedding_manager(self) -> EmbeddingManager:
        if self._embedding_manager is None:
            self._embedding_manager = get_embedding_manager()
        return self._embedding_manager
    
    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            hybrid_config = HybridConfig(
                vector_top_k=self.config.vector_top_k,
                fts_top_k=self.config.fts_top_k,
                graph_top_k=self.config.graph_top_k,
                enable_rerank=self.config.enable_rerank,
                rerank_top_k=self.config.rerank_top_k
            )
            self._retriever = HybridRetriever(
                config=hybrid_config,
                embedding_manager=self.embedding_manager
            )
        return self._retriever
    
    @property
    def context_builder(self) -> ContextBuilder:
        if self._context_builder is None:
            context_config = ContextConfig(
                max_tokens=self.config.max_context_tokens,
                include_source=self.config.include_source
            )
            self._context_builder = ContextBuilder(context_config)
        return self._context_builder
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True
    ) -> list[Union[RerankResult, RetrievalResult]]:
        """
        执行检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            use_rerank: 是否使用精排
            
        Returns:
            检索结果列表
        """
        return self.retriever.search(query, top_k=top_k, use_rerank=use_rerank)
    
    def retrieve_and_build_context(
        self,
        query: str,
        top_k: int = 5,
        template: str = "qa",
        extra_context: Optional[str] = None
    ) -> tuple[str, list]:
        """
        检索并构建上下文
        
        Args:
            query: 查询文本
            top_k: 检索数量
            template: Prompt 模板名称
            extra_context: 额外上下文
            
        Returns:
            (构建好的 Prompt, 检索结果列表)
        """
        results = self.retrieve(query, top_k=top_k)
        prompt = self.context_builder.build(
            query=query,
            results=results,
            template_name=template,
            extra_context=extra_context
        )
        return prompt, results
    
    def get_context(
        self,
        query: str,
        top_k: int = 5
    ) -> str:
        """
        获取检索上下文（不含模板）
        
        用于注入到其他系统的 Prompt 中
        """
        results = self.retrieve(query, top_k=top_k)
        return self.context_builder.build_context_only(results)
    
    def cleanup(self):
        """释放资源"""
        if self._embedding_manager:
            self._embedding_manager.unload()


# 全局单例
_rag_instance: Optional[RAG] = None


def get_rag(**kwargs) -> RAG:
    """获取全局 RAG 实例"""
    global _rag_instance
    if _rag_instance is None:
        config = RAGConfig(**kwargs) if kwargs else None
        _rag_instance = RAG(config)
    return _rag_instance


# 快捷函数
def retrieve(query: str, top_k: int = 5, **kwargs) -> list:
    """快捷检索"""
    return get_rag(**kwargs).retrieve(query, top_k=top_k)


def get_context(query: str, top_k: int = 5, **kwargs) -> str:
    """快捷获取上下文"""
    return get_rag(**kwargs).get_context(query, top_k=top_k)


def retrieve_and_build_prompt(
    query: str,
    top_k: int = 5,
    template: str = "qa",
    **kwargs
) -> str:
    """快捷检索+构建 Prompt"""
    prompt, _ = get_rag(**kwargs).retrieve_and_build_context(
        query, top_k=top_k, template=template
    )
    return prompt
