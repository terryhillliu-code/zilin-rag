"""
Query 重写器 (v5.2)
- 使用 LLM 将用户原始口语转化为更适合 RAG 检索的语义查询
- 支持 HyDE (Hypothetical Document Embeddings) 模式
"""
import json
import requests
from typing import List, Optional

class QueryRewriter:
    """Query 重写器"""

    def __init__(self, api_url: str = "http://127.0.0.1:8765/chat"):
        self.api_url = api_url

    def rewrite(self, query: str, mode: str = "balanced") -> List[str]:
        """
        重写查询
        
        Args:
            query: 原始查询
            mode: "precise" (1个精准) | "broad" (3个多维) | "hyde" (1个虚构答案)
        
        Returns:
            重写后的查询列表
        """
        prompt = ""
        if mode == "broad":
            prompt = f"针对以下不完整的查询，生成 3 个不同维度的学术性 RAG 检索词。直接返回 JSON 列表：[\"q1\", \"q2\", \"q3\"]\n查询：{query}"
        elif mode == "hyde":
            prompt = f"请为一个科学研究 RAG 系统生成一个针对以下问题的简短、虚构但合理的理想答案（约 100 字）。我们不需要真实性，只需要语义上的丰富。问题：{query}"
        else:
            prompt = f"请将以下用户口语化的查询转换为一个更适合在学术论文库中检索的精准术语。直接返回结果，不要解释。查询：{query}"

        try:
            resp = requests.post(
                self.api_url,
                json={
                    "message": prompt,
                    "model": "qwen3.5-plus",
                    "temperature": 0.3
                },
                timeout=30
            )
            
            if resp.status_code == 200:
                result = resp.json().get("response", query)
                
                # 尝试解析 JSON 列表
                if mode == "broad":
                    try:
                        # 找 JSON 块
                        import re
                        json_match = re.search(r"(\[.*\])", result, re.DOTALL)
                        if json_match:
                            return json.loads(json_match.group(1))
                    except:
                        return [result]
                
                return [result]
        except Exception as e:
            print(f"[Rewriter] 错误: {e}")
            
        return [query]

# 快捷调用
def rewrite_query(query: str, mode: str = "precise") -> List[str]:
    rewriter = QueryRewriter()
    return rewriter.rewrite(query, mode=mode)
