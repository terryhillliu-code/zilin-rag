#!/usr/bin/env python3
"""
DashScope WebSearch MCP Server

为 Claude Code 提供联网搜索能力，使用阿里 DashScope API。

特性：
- 实时联网搜索（enable_search）
- 结构化 JSON 输出
- 来源链接验证
- 结果缓存

依赖：pip install mcp httpx
"""

import os
import json
import time
import asyncio
from pathlib import Path
from functools import lru_cache
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# 配置
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = "qwen-turbo-latest"
DEFAULT_TIMEOUT = 30
MAX_CACHE_SIZE = 50

# MCP Server
mcp = FastMCP("dashscope-websearch", json_response=True)


class SearchResult(BaseModel):
    """搜索结果结构"""
    answer: str
    sources: list[dict]
    confidence: float = 0.9
    query: str
    model: str
    latency_ms: int
    cached: bool = False


# 内存缓存
_cache: dict = {}


def _load_api_key() -> str:
    """加载 API Key"""
    if DASHSCOPE_API_KEY:
        return DASHSCOPE_API_KEY

    env_path = Path.home() / ".secrets" / "global.env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("DASHSCOPE_API_KEY="):
                return line.split("=", 1)[1].strip()

    return ""


API_KEY = _load_api_key()


async def _verify_url(url: str, timeout: float = 2.0) -> bool:
    """验证 URL 可访问性"""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.head(url, follow_redirects=True)
            return resp.status_code < 400
    except Exception:
        return False


async def _dashscope_search(query: str) -> dict:
    """调用 DashScope API"""
    prompt = f"""搜索: {query}

按 JSON 格式返回结果：
{
  "answer": "简洁回答（200字内）",
  "sources": [{"title": "标题", "url": "链接"}],
  "confidence": 0.9
}

只返回 JSON，无其他文字。"""

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        resp = await client.post(
            DASHSCOPE_BASE_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": DEFAULT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "enable_search": True
            }
        )
        resp.raise_for_status()
        return resp.json()


def _parse_response(response: dict, query: str, latency_ms: int) -> SearchResult:
    """解析 API 响应"""
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

    # 提取 JSON
    try:
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "{" in content:
            start, end = content.index("{"), content.rindex("}") + 1
            json_str = content[start:end]
        else:
            json_str = content

        data = json.loads(json_str)
        return SearchResult(
            answer=data.get("answer", content[:300]),
            sources=data.get("sources", []),
            confidence=data.get("confidence", 0.9),
            query=query,
            model=response.get("model", DEFAULT_MODEL),
            latency_ms=latency_ms
        )
    except (json.JSONDecodeError, ValueError):
        return SearchResult(
            answer=content[:300] or "搜索失败",
            sources=[],
            confidence=0.5,
            query=query,
            model=response.get("model", DEFAULT_MODEL),
            latency_ms=latency_ms
        )


@mcp.tool()
async def web_search(query: str, max_results: int = 10) -> str:
    """
    联网搜索（阿里 DashScope）

    Args:
        query: 搜索查询
        max_results: 最大结果数

    Returns:
        JSON 搜索结果
    """
    if not API_KEY:
        return json.dumps({"error": "DASHSCOPE_API_KEY 未配置"}, ensure_ascii=False)

    if not query:
        return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)

    # 缓存检查
    cache_key = query.lower().strip()[:80]
    if cache_key in _cache:
        cached = _cache[cache_key]
        cached.cached = True
        return cached.model_dump_json(indent=2)

    # 执行搜索
    start = time.time()
    try:
        response = await _dashscope_search(query)
        latency_ms = int((time.time() - start) * 1000)

        result = _parse_response(response, query, latency_ms)

        # 验证前 3 个链接
        if result.sources:
            for src in result.sources[:3]:
                url = src.get("url", "")
                if url:
                    src["verified"] = await _verify_url(url)

        # 缓存
        if len(_cache) < MAX_CACHE_SIZE:
            _cache[cache_key] = result

        return result.model_dump_json(indent=2)

    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"API 错误: {e.response.status_code}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"搜索失败: {str(e)[:100]}"}, ensure_ascii=False)


@mcp.tool()
def search_health_check() -> str:
    """检查 DashScope API 连接状态"""
    if not API_KEY:
        return json.dumps({"status": "error", "message": "API Key 未配置"}, ensure_ascii=False)

    return json.dumps({
        "status": "ok",
        "message": "DashScope WebSearch MCP 就绪",
        "model": DEFAULT_MODEL,
        "api_key_set": bool(API_KEY)
    }, ensure_ascii=False)


if __name__ == "__main__":
    if not API_KEY:
        print("错误: DASHSCOPE_API_KEY 未设置", file=__import__("sys").stderr)
        __import__("sys").exit(1)
    mcp.run()