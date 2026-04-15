"""
知微 MCP Server
提供 RAG 检索、网络搜索和系统状态查询工具

使用 FastMCP 实现 (官方 Python SDK)
参考：https://github.com/modelcontextprotocol/python-sdk

工具列表 (8个):
- search_knowledge: 三轨检索（本地知识库）
- web_search: 多后端聚合搜索（Exa → Tavily → DDGS）
- WebSearch: web_search 的人类友好格式
- web_search_status: API 用量查询
- search_diagnostics: 搜索性能诊断
- search_social: 社交平台深度搜索（last30days v3 引擎）
- get_system_health: 系统+Docker状态
- get_recent_changes: CHANGELOG变更
- get_task_queue: 开发任务队列
- get_vectorize_status: 向量化进度

网络搜索说明:
- 单入口聚合：Exa (语义) → Tavily (AI优化) → DDGS (零成本保底)
- API Key 配置在 ~/.secrets/global.env
- 内置月度用量追踪，防止超额
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# 设置离线模式，避免 HuggingFace Hub 网络请求
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
    多端聚合搜索，自动降级 + 用量追踪

    降级链: Exa (语义) → Tavily (AI优化) → DDGS (DuckDuckGo, 零成本)

    Args:
        query: 搜索关键词
        count: 返回结果数量，默认 5

    Returns:
        JSON 格式的搜索结果
    """
    from search.search_multi import search as multi_search

    result = multi_search(query, count)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================================
# WebSearch 别名 - 覆盖内置工具
# ============================================================================

@mcp.tool()
def WebSearch(query: str) -> str:
    """
    网络搜索工具（覆盖 Claude Code 内置 WebSearch）

    多后端聚合搜索：Exa → Tavily → DDGS，带 5 分钟本地缓存

    Args:
        query: 搜索关键词

    Returns:
        搜索结果，包含 URL 列表和摘要
    """
    result = web_search(query, count=5)

    try:
        data = json.loads(result)
        if "error" in data:
            diag = data.get("diagnostics", "")
            return f"搜索失败: {data['error']}\n诊断: {diag}"

        provider = data.get('provider', 'unknown')
        cached_mark = " [缓存]" if data.get("_cached") else ""
        output = f"搜索: {query}\n\n"
        output += f"来源: {provider}{cached_mark}\n\n"

        results = data.get("results", [])
        if results:
            output += "结果:\n"
            for r in results:
                title = r.get("title", "链接")
                url = r.get("url", "")
                output += f"- [{title}]({url})\n"
                snippet = r.get("snippet", "")
                if snippet:
                    output += f"  {snippet[:150]}\n"

        return output
    except json.JSONDecodeError:
        return result


@mcp.tool()
def web_search_status() -> str:
    """
    查询搜索 API 用量状态

    Returns:
        各 provider 的已用/限额/剩余
    """
    from search.search_multi import get_quota_status

    status = get_quota_status()
    lines = ["API 搜索用量统计 (本月)", ""]
    for provider, info in status.items():
        used = info["used"]
        limit = info["limit"]
        avail = "✅" if info["available"] else "❌ 已超额"
        if limit == "unlimited":
            lines.append(f"  {provider}: {used} 次 (无限) {avail}")
        else:
            remaining = info["remaining"]
            lines.append(f"  {provider}: {used}/{limit} (剩余 {remaining}) {avail}")
    return "\n".join(lines)


@mcp.tool()
def search_diagnostics() -> str:
    """
    搜索性能诊断报告

    返回最近 10 次搜索的详细记录：provider、耗时、成功/失败原因、缓存命中
    """
    import json
    from search.search_multi import get_diagnostics

    entries = get_diagnostics()
    if not entries:
        return "暂无诊断记录"

    lines = ["搜索性能诊断 (最近 10 次)", "=" * 40, ""]
    for e in entries:
        status_icon = {"success": "✅", "error": "❌", "cache_hit": "⚡"}.get(e["status"], "?")
        lines.append(f"{status_icon} {e['provider']} | {e['elapsed_s']}s | {e['time']}")
        if e.get("error"):
            lines.append(f"   错误: {e['error']}")
        lines.append("")

    return "\n".join(lines)


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
def search_social(query: str, sources: str = "reddit,hn,github", depth: str = "quick") -> str:
    """
    社交平台深度搜索（last30days v3 引擎）

    搜索 Reddit、Hacker News、GitHub 等平台的最近 30 天内容，
    按真实用户互动度（upvote/like/押注金额）评分并合成报告。

    免费源（无需 API Key）：reddit、hn（Hacker News）、polymarket、github
    可选源：x（Twitter）、youtube、tiktok、threads、pinterest、bluesky

    Args:
        query: 搜索关键词
        sources: 逗号分隔的源列表，默认 reddit,hn,github
        depth: 搜索深度，quick(快速) / normal / deep

    Returns:
        JSON 格式的搜索报告，含排名聚类、Best Takes 等
    """
    skill_dir = Path.home() / "last30days-skill"
    script = skill_dir / "scripts" / "last30days.py"

    if not script.exists():
        return json.dumps({"error": "last30days-skill 未安装，请运行 git clone 到 ~/last30days-skill"}, ensure_ascii=False)

    cmd = [sys.executable, str(script), query, f"--search={sources}"]
    if depth == "deep":
        cmd.append("--deep")
    elif depth == "quick":
        cmd.append("--quick")
    # depth="normal" 不传 flag，走脚本默认行为

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            env={**os.environ}
        )
        if result.returncode == 0:
            # 清理 stdout 中可能的 Python 警告行，只返回 JSON
            output = result.stdout.strip()
            lines = output.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('{') or line.startswith('['):
                    return '\n'.join(lines[i:])
            return output
        stderr = result.stderr.strip()
        if stderr:
            return json.dumps({"error": stderr[-500:]}, ensure_ascii=False)
        return json.dumps({"error": f"搜索失败，退出码 {result.returncode}"}, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "搜索超时 (5分钟限制)，建议使用 --search 缩小搜索范围"}, ensure_ascii=False)
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