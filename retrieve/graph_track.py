"""
轨道 C：LightRAG 图谱检索
通过子进程调用，避免 asyncio 冲突
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from retrieve.vector_track import RetrievalResult


class GraphTrack:
    """图谱检索轨道（子进程隔离）"""

    # 默认超时设为 15s（OPT-002 更新）
    # 原因：120s 超时过长拖慢整体检索，15s 内未返回则跳过图谱轨道
    DEFAULT_TIMEOUT = 15

    def __init__(
        self,
        graph_db_path: str = "~/zhiwei-scheduler/graph_db",
        cli_script: str = "~/zhiwei-scheduler/graph_query_cli.py",
        timeout: int = None
    ):
        self.graph_db_path = os.path.expanduser(graph_db_path)
        self.cli_script = os.path.expanduser(cli_script)
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        
        # 强制指定 zhiwei-scheduler 的 venv 路径，以确保找到 lightrag 依赖
        self.venv_python = os.path.expanduser("~/zhiwei-scheduler/venv/bin/python3")
        if not os.path.exists(self.venv_python):
            self.venv_python = sys.executable # 保持回退
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid"
    ) -> list[RetrievalResult]:
        """
        图谱检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            mode: 检索模式 (local/global/hybrid)
            
        Returns:
            检索结果列表
        """
        # 检查依赖文件
        if not os.path.exists(self.cli_script):
            print(f"[GraphTrack] CLI 脚本不存在: {self.cli_script}", file=sys.stderr)
            return self._fallback_search(query, top_k)
        
        if not os.path.exists(self.graph_db_path):
            print(f"[GraphTrack] 图数据库不存在: {self.graph_db_path}", file=sys.stderr)
            return []
        
        try:
            # 通过子进程调用，避免 asyncio 冲突
            result = subprocess.run(
                [
                    self.venv_python,
                    self.cli_script,
                    "--query", query,
                    "--mode", mode,
                    "--top-k", str(top_k),
                    "--output", "json"
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=os.path.dirname(self.cli_script)
            )
            
            if result.returncode != 0:
                print(f"[GraphTrack] CLI 错误: {result.stderr}", file=sys.stderr)
                return self._fallback_search(query, top_k)
            
            # 解析 JSON 输出
            output = json.loads(result.stdout)
            
            results = []
            for item in output.get('results', [])[:top_k]:
                results.append(RetrievalResult(
                    text=item.get('content', ''),
                    raw_text=item.get('content', ''),
                    source=item.get('source', 'graph'),
                    score=item.get('score', 0.5),
                    track='graph',
                    metadata={
                        'entities': item.get('entities', []),
                        'relations': item.get('relations', []),
                        'community': item.get('community', '')
                    }
                ))
            
            return results
            
        except subprocess.TimeoutExpired:
            print(f"[GraphTrack] 查询超时 ({self.timeout}s)", file=sys.stderr)
            return []
        except json.JSONDecodeError as e:
            print(f"[GraphTrack] JSON 解析错误: {e}", file=sys.stderr)
            return self._fallback_search(query, top_k)
        except Exception as e:
            print(f"[GraphTrack] 查询错误: {e}", file=sys.stderr)
            return []
    
    def _fallback_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """
        降级方案：直接读取图数据库的实体文件
        """
        entities_file = Path(self.graph_db_path) / "entities.json"
        if not entities_file.exists():
            return []
        
        try:
            with open(entities_file) as f:
                entities = json.load(f)
            
            # 简单关键词匹配
            query_lower = query.lower()
            matched = []
            
            for entity in entities:
                name = entity.get('name', '').lower()
                desc = entity.get('description', '').lower()
                
                if query_lower in name or query_lower in desc:
                    matched.append(RetrievalResult(
                        text=f"实体: {entity.get('name')}\n描述: {entity.get('description', '')}",
                        raw_text=entity.get('description', ''),
                        source='graph_fallback',
                        score=0.3,
                        track='graph',
                        metadata={'entity_name': entity.get('name')}
                    ))
            
            return matched[:top_k]
            
        except Exception as e:
            print(f"[GraphTrack] 降级检索错误: {e}", file=sys.stderr)
            return []
