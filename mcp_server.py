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

    优先级（智谱优先）：
    1. 智谱 GLM web_search（免费，首选）
    2. GitHub 搜索（免费，适合技术仓库）
    3. 阿里通义深度研究（$0.0004/次，备用）
    4. Perplexity sonar（$0.005/次，备用）

    Args:
        query: 搜索关键词
        count: 返回结果数量，默认 5

    Returns:
        JSON 格式的搜索结果
    """
    import os
    import urllib.request
    import urllib.parse
    import re

    # 获取配置
    zhipu_key = ""
    gh_token = ""
    api_key = ""
    env_path = Path.home() / ".secrets" / "global.env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ZHIPU_API_KEY="):
                zhipu_key = line.split("=")[1].strip()
            if line.startswith("GH_TOKEN="):
                gh_token = line.split("=")[1].strip()
            if line.startswith("OPENROUTER_API_KEY="):
                api_key = line.split("=")[1].strip()

    # 优先级 1: 智谱 GLM web_search（免费，首选，30秒超时）
    if zhipu_key:
        result = _search_zhipu(query, count, zhipu_key)
        # 成功且有结果才返回，否则继续 fallback
        if "error" not in result and result.get("count", 0) > 0:
            return json.dumps(result, ensure_ascii=False, indent=2)

    # 优先级 2: GitHub 搜索（免费，适合技术内容）
    if gh_token and _is_technical_query(query):
        result = _search_github(query, count, gh_token)
        if "error" not in result and result.get("count", 0) > 0:
            return json.dumps(result, ensure_ascii=False, indent=2)

    # 优先级 3: 阿里通义深度研究（备用）
    if api_key:
        result = _search_tongyi_deep(query, count, api_key)
        if "error" not in result and result.get("count", 0) > 0:
            return json.dumps(result, ensure_ascii=False, indent=2)

    # 优先级 4: Perplexity 备用
    if api_key:
        result = _search_perplexity(query, count, api_key)
        if "error" not in result and result.get("count", 0) > 0:
            return json.dumps(result, ensure_ascii=False, indent=2)

    return json.dumps({"error": "所有搜索方案失败", "attempted": ["zhipu", "github", "tongyi", "perplexity"]}, ensure_ascii=False, indent=2)


def _search_zhipu(query: str, count: int, api_key: str) -> dict:
    """智谱 GLM web_search（首选，免费）

    使用 subprocess + curl 调用，避免 urllib 超时问题。
    """
    import subprocess
    import re
    import time

    start_time = time.time()
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    data = {
        "model": "glm-4.5",
        "messages": [{"role": "user", "content": query}],
        "tools": [{"type": "web_search", "web_search": {"search_result": True}}],
        "stream": False
    }

    try:
        # 使用 curl 调用 API，30秒超时以加快 fallback
        result = subprocess.run(
            [
                "curl", "-s", "-m", "30",
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {api_key}",
                "-d", json.dumps(data),
                url
            ],
            capture_output=True,
            text=True,
            timeout=35
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            return {"error": f"curl 失败 (耗时{elapsed:.1f}s): {result.stderr[:100]}"}

        response = json.loads(result.stdout)

        # 检查 API 返回是否有错误
        if "error" in response:
            return {"error": f"智谱 API 错误: {response['error']}"}

        message = response["choices"][0]["message"]
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")

        # 提取 URL 作为参考来源
        urls = re.findall(r'https?://[^\s\)\]\>\"]+', content)

        results = []
        for i, url_item in enumerate(urls[:count]):
            results.append({
                "rank": i + 1,
                "url": url_item,
                "source": "zhipu-glm-web_search"
            })

        return {
            "query": query,
            "method": "zhipu-glm-4.5-web_search (免费)",
            "count": len(results),
            "results": results,
            "answer": content[:1000],
            "reasoning": reasoning[:300] if reasoning else None,
            "model": response.get("model", "glm-4.5"),
            "usage": response.get("usage", {}),
            "elapsed_seconds": round(elapsed, 1)
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {"error": f"智谱搜索超时 (耗时{elapsed:.1f}s，自动fallback)"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON 解析失败: {str(e)[:50]}"}
    except Exception as e:
        elapsed = time.time() - start_time
        return {"error": f"智谱搜索失败 (耗时{elapsed:.1f}s): {str(e)[:50]}"}


def _is_technical_query(query: str) -> bool:
    """判断是否是技术查询（适合 GitHub 搜索）"""
    tech_keywords = ["github", "repo", "library", "package", "npm", "pip", "git", "code", "api", "sdk", "cli", "framework"]
    return any(kw in query.lower() for kw in tech_keywords)


def _search_tongyi_deep(query: str, count: int, api_key: str) -> dict:
    """阿里通义深度研究（最便宜，$0.0004/次）"""
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data = {
            "model": "alibaba/tongyi-deepresearch-30b-a3b",
            "messages": [{"role": "user", "content": f"搜索: {query}\n列出 {count} 个最相关的结果，包含标题和链接。"}]
        }

        req = urllib.request.Request(url, json.dumps(data).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            response = json.loads(resp.read().decode())

        content = response["choices"][0]["message"]["content"]

        # 提取 URL
        urls = re.findall(r'https?://[^\s\)\]\>\"]+', content)

        results = []
        for i, url in enumerate(urls[:count]):
            results.append({
                "rank": i + 1,
                "url": url,
                "source": "tongyi-deepresearch"
            })

        return {
            "query": query,
            "method": "tongyi-deepresearch (阿里通义)",
            "count": len(results),
            "results": results,
            "cost_usd": response.get("usage", {}).get("cost", 0),
            "full_response": content[:500]
        }

    except Exception as e:
        return {"error": f"通义搜索失败: {str(e)}"}


def _search_github(query: str, count: int, token: str) -> dict:
    """GitHub 仓库搜索（完全免费）"""
    try:
        url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&per_page={count}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json"
        }

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            response = json.loads(resp.read().decode())

        results = []
        for item in response.get("items", [])[:count]:
            results.append({
                "rank": len(results) + 1,
                "title": item.get("full_name", ""),
                "url": item.get("html_url", ""),
                "description": (item.get("description") or "")[:100],
                "stars": item.get("stargazers_count", 0),
                "language": item.get("language", ""),
                "source": "github"
            })

        return {
            "query": query,
            "method": "github (免费)",
            "total_count": response.get("total_count", 0),
            "count": len(results),
            "results": results
        }

    except Exception as e:
        return {"error": f"GitHub 搜索失败: {str(e)}"}


def _search_perplexity(query: str, count: int, api_key: str) -> dict:
    """使用 OpenRouter Perplexity 搜索"""
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/zhiwei-bot"
        }
        data = {
            "model": "perplexity/sonar",
            "messages": [{"role": "user", "content": f"搜索: {query}\n列出 {count} 个相关结果，包含标题和URL。"}],
            "max_tokens": 500
        }

        req = urllib.request.Request(url, json.dumps(data).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            response = json.loads(resp.read().decode())

        content = response["choices"][0]["message"]["content"]

        # 提取 URL
        import re
        urls = re.findall(r'https?://[^\s\)\]\>\"]+', content)

        results = []
        for i, url in enumerate(urls[:count]):
            # 提取标题（URL 前的文字）
            title_match = re.search(rf'([^\n]+)\s*{re.escape(url)}', content)
            title = title_match.group(1).strip() if title_match else f"结果 {i+1}"
            results.append({
                "rank": i + 1,
                "title": title[:80],
                "url": url,
                "source": "perplexity"
            })

        return {
            "query": query,
            "method": "perplexity/sonar",
            "count": len(results),
            "results": results,
            "cost_usd": response.get("usage", {}).get("cost", 0)
        }

    except Exception as e:
        return {"error": f"Perplexity 搜索失败: {str(e)}"}


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