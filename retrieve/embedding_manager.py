"""
本地 Embedding 管理器
- MPS 加速（Apple Silicon）
- 按需加载，首次约 3-5 秒
- 空闲 30 分钟自动释放内存（OPT-009 优化）

OPT-009 优化说明：
- 默认 idle_timeout 从 300s 提升至 1800s（30分钟）
- 工作时段减少模型重复加载次数
- 夜间仍会自动释放，避免内存压力
"""
import gc
import time
import sys
import threading
from typing import Optional
import numpy as np


class EmbeddingManager:
    """线程安全的 Embedding 模型管理器"""

    def __init__(
        self,
        model_name: str = 'BAAI/bge-large-zh-v1.5',
        device: str = 'mps',
        idle_timeout: int = 1800,  # OPT-009: 30分钟，减少重复加载
        normalize: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.idle_timeout = idle_timeout
        self.normalize = normalize
        
        self._model = None
        self._last_used = 0.0
        self._lock = threading.Lock()
        self._cleanup_timer: Optional[threading.Timer] = None
        self._loading = False
    
    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度条
            
        Returns:
            numpy 数组，shape = (len(texts), dimension)
        """
        with self._lock:
            self._ensure_loaded()
            self._last_used = time.time()
            self._schedule_cleanup()
        
        # 编码（在锁外执行，避免长时间持锁）
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            device=self.device
        )
        
        return np.array(embeddings)
    
    def encode_single(self, text: str) -> np.ndarray:
        """编码单条文本"""
        return self.encode([text])[0]
    
    @property
    def dimension(self) -> int:
        """向量维度"""
        with self._lock:
            self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()
    
    @property
    def is_loaded(self) -> bool:
        """模型是否已加载"""
        return self._model is not None
    
    def preload(self):
        """预加载模型（可选，用于系统启动时预热）"""
        with self._lock:
            self._ensure_loaded()
    
    def unload(self):
        """手动卸载模型"""
        with self._lock:
            self._do_unload()
    
    def _ensure_loaded(self):
        """确保模型已加载（需在锁内调用）"""
        if self._model is not None:
            return
        
        if self._loading:
            return
        
        self._loading = True
        try:
            print(f"[EmbeddingManager] 加载模型: {self.model_name} (device={self.device})", file=sys.stderr)
            start = time.time()
            
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            elapsed = time.time() - start
            print(f"[EmbeddingManager] 模型加载完成，耗时 {elapsed:.1f}s", file=sys.stderr)
        finally:
            self._loading = False
    
    def _schedule_cleanup(self):
        """安排空闲清理（需在锁内调用）"""
        # OPT-009: idle_timeout <= 0 表示常驻模式，不自动卸载
        if self.idle_timeout <= 0:
            return

        if self._cleanup_timer:
            self._cleanup_timer.cancel()

        self._cleanup_timer = threading.Timer(
            self.idle_timeout,
            self._try_unload
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
    
    def _try_unload(self):
        """尝试卸载（定时器回调）"""
        with self._lock:
            if time.time() - self._last_used >= self.idle_timeout:
                self._do_unload()
    
    def _do_unload(self):
        """执行卸载（需在锁内调用）"""
        if self._model is None:
            return
        
        print("[EmbeddingManager] 空闲超时，释放模型内存", file=sys.stderr)
        
        del self._model
        self._model = None
        
        gc.collect()
        
        # 清理 MPS 缓存
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None


# 全局单例（可选）
_default_manager: Optional[EmbeddingManager] = None


def get_embedding_manager(**kwargs) -> EmbeddingManager:
    """获取全局 EmbeddingManager 实例"""
    global _default_manager
    if _default_manager is None:
        _default_manager = EmbeddingManager(**kwargs)
    return _default_manager


def encode(texts: list[str], **kwargs) -> np.ndarray:
    """快捷编码函数"""
    return get_embedding_manager().encode(texts, **kwargs)
