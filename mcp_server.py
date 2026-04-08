"""
知微 MCP Server
提供 RAG 检索、网络搜索和系统状态查询工具

使用 FastMCP 实现 (官方 Python SDK)
参考：https://github.com/modelcontextprotocol/python-sdk

工具列表 (6个):
- search_knowledge: 三轨检索（本地知识库）
- web_search: 网络搜索（OpenRouter Perplexity，已有 key）
- get_system_health: 系统+Docker状态
- get_recent_changes: CHANGELOG变更
- get_task_queue: 开发任务队列
- get_vectorize_status: 向量化进度

网络搜索说明:
- 使用 OpenRouter 的 Perplexity sonar 模型
- 自动读取 ~/.secrets/openrouter_api_key.txt
- 无需额外配置 API key
"""

import subprocess
import json
import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# 创建 MCP Server 实例
mcp = FastMCP("zhiwei-mcp", json_response=True)


@mcp.tool()
def search_knowledge(query: str, top_k: int = 5) -> str:
    """
    搜索知识库，返回相关文档片段

    Args:
        query: 搜索查询文本
        top_k: 返回结果数量，默认 5
    """
    try:
        # 直接导入 api 模块调用 retrieve
        sys.path.insert(0, str(Path(__file__).parent))
        from api import retrieve

        results = retrieve(query, top_k=top_k)

        if not results:
            return json.dumps({"status": "success", "message": "未找到相关结果"}, ensure_ascii=False)

        formatted = []
        for i, r in enumerate(results, 1):
            # 兼容 RerankResult 和 RetrievalResult
            text = getattr(r, 'text', '') or getattr(r, 'raw_text', '')
            score = getattr(r, 'score', 0) or getattr(r, '_distance', 0)
            source = getattr(r, 'source', 'unknown')
            track = getattr(r, 'track', 'unknown')

            formatted.append({
                "rank": i,
                "text": text[:500] if text else "",
                "score": round(score, 4) if score else 0,
                "source": source,
                "track": track
            })

        return json.dumps({
            "query": query,
            "count": len(formatted),
            "results": formatted
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def web_search(query: str, count: int = 5) -> str:
    """
    网络搜索（替代 Claude Code 内置 WebSearch）

    使用 OpenRouter Perplexity 搜索模型，无需额外配置 API key。

    Args:
        query: 搜索关键词
        count: 返回结果数量，默认 5

    Returns:
        JSON 格式的搜索结果，包含标题、URL、摘要
    """
    import os
    import urllib.request
    import urllib.parse

    # 从配置文件读取 OpenRouter API key
    api_key = ""

    # 优先读取 global.env
    env_path = Path.home() / ".secrets" / "global.env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                api_key = line.split("=")[1].strip()
                break

    # 备用：读取独立 key 文件
    if not api_key:
        key_path = Path.home() / ".secrets" / "openrouter_api_key.txt"
        if key_path.exists():
            api_key = key_path.read_text().strip()

    # 备用：环境变量
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key:
        return json.dumps({
            "error": "未配置 OPENROUTER_API_KEY",
            "hint": "OpenRouter API key 文件不存在: ~/.secrets/openrouter_api_key.txt"
        }, ensure_ascii=False, indent=2)

    try:
        # 使用 OpenRouter Perplexity 搜索模型
        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/zhiwei-bot",
            "X-Title": "Zhiwei Web Search"
        }

        data = {
            "model": "perplexity/sonar",  # 轻量搜索模型
            "messages": [
                {
                    "role": "user",
                    "content": f"搜索: {query}\n\n请列出 {count} 个最相关的搜索结果，每个结果包含标题、URL 和简短摘要。格式为 JSON 列表。"
                }
            ],
            "max_tokens": 1000
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode(),
            headers=headers,
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            response = json.loads(resp.read().decode())

        # 解析 Perplexity 响应
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # 尝试提取 JSON 格式的结果
        # Perplexity 返回的是文本，可能包含 JSON 或 markdown 格式的链接
        import re

        # 提取 URL
        urls = re.findall(r'https?://[^\s\)\]\>]+', content)

        results = []
        for i, url in enumerate(urls[:count]):
            # 尝试提取标题（URL 前面的文字）
            results.append({
                "rank": i + 1,
                "url": url,
                "content": content[:500]
            })

        return json.dumps({
            "query": query,
            "model": "perplexity/sonar",
            "count": len(results),
            "results": results,
            "full_response": content[:1000]
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
@mcp.tool()
def get_system_health() -> str:
    """获取知微系统健康状态，包括服务和 Docker 容器状态"""
    try:
        sys.path.insert(0, str(Path.home() / "zhiwei-bot"))
        from core.health_check import get_system_health_dict
        status = get_system_health_dict()
        return json.dumps(status, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)



@mcp.tool()
def get_recent_changes(days: int = 3) -> str:
    """
    获取最近的系统变更记录

    Args:
        days: 查询最近几天的变更，默认 3 天
    """
    import json
    from pathlib import Path

    changelog_path = Path.home() / "zhiwei-docs" / "CHANGELOG.md"

    if not changelog_path.exists():
        return "错误: CHANGELOG.md 不存在"

    try:
        content = changelog_path.read_text()
        lines = content.split("\n")

        # 简单提取最近的日期块
        recent = []
        capture = False

        for line in lines:
            if line.startswith("### 2026-"):
                recent.append(line)
                capture = True
            elif capture and line.startswith("---"):
                break
            elif capture:
                recent.append(line)

        result = "\n".join(recent[:50])  # 限制行数
        return result if result else "无最近变更记录"

    except Exception as e:
        return f"错误: {str(e)}"


@mcp.tool()
def get_task_queue(limit: int = 10) -> str:
    """
    获取 zhiwei-dev 任务队列状态

    Args:
        limit: 返回最近任务数量，默认 10
    """
    import sqlite3

    db_path = Path.home() / "zhiwei-dev" / "tasks.db"

    if not db_path.exists():
        return json.dumps({"error": "tasks.db 不存在"}, ensure_ascii=False)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 统计各状态数量
        cursor.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status")
        status_counts = dict(cursor.fetchall())

        # 获取最近任务
        cursor.execute("""
            SELECT id, status, input, created_at, backend, error
            FROM tasks
            ORDER BY id DESC LIMIT ?
        """, (limit,))

        tasks = []
        for row in cursor.fetchall():
            tasks.append({
                "id": row[0],
                "status": row[1],
                "input": row[2][:100] if row[2] else "",
                "created_at": row[3],
                "backend": row[4],
                "error": row[5][:200] if row[5] else None
            })

        conn.close()

        return json.dumps({
            "status_counts": status_counts,
            "recent_tasks": tasks
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def get_vectorize_status() -> str:
    """获取知识库向量化状态"""
    try:
        # 导入 LanceStore 获取向量数
        sys.path.insert(0, str(Path(__file__).parent))
        from ingest.lance_store import LanceStore

        store = LanceStore()
        vector_count = store.count()

        # 获取数据目录大小
        data_dir = Path.home() / "zhiwei-rag" / "data"
        db_size = 0
        if data_dir.exists():
            for f in data_dir.rglob("*"):
                if f.is_file():
                    db_size += f.stat().st_size

        return json.dumps({
            "vector_count": vector_count,
            "db_size_mb": round(db_size / 1024 / 1024, 2),
            "lancedb_path": str(data_dir / "lance_db")
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()