"""
知微 MCP Server
提供 RAG 检索和系统状态查询工具

使用 FastMCP 实现 (官方 Python SDK)
参考：https://github.com/modelcontextprotocol/python-sdk

工具列表 (5个):
- search_knowledge: 三轨检索
- get_system_health: 系统+Docker状态
- get_recent_changes: CHANGELOG变更
- get_task_queue: 开发任务队列
- get_vectorize_status: 向量化进度
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
def get_system_health() -> str:
    """获取知微系统健康状态，包括服务和 Docker 容器状态"""
    import json

    status = {
        "services": {},
        "docker": {}
    }

    # 检查 launchd 服务
    services = [
        "com.zhiwei.bot",
        "com.zhiwei.scheduler",
        "com.zhiwei.dev-worker",
        "com.zhiwei.dispatcher",
        "com.zhiwei.rag-api"
    ]

    try:
        result = subprocess.run(
            ["launchctl", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 3:
                    pid, last_exit_code, name = parts[0], parts[1], parts[2]
                    if name in services:
                        # 使用 PID 存在性判断运行状态
                        # launchctl list 格式: PID 上次退出码 服务名
                        # PID 不是 "-" 说明进程正在运行
                        is_running = pid != "-"

                        if name == "com.zhiwei.rag-api":
                            # rag-api 需要额外检查 health 接口
                            if not is_running:
                                status["services"][name] = {
                                    "status": "unhealthy",
                                    "pid": pid
                                }
                            else:
                                # 进程存在，检查 health 接口
                                try:
                                    import requests
                                    resp = requests.get("http://127.0.0.1:8765/health", timeout=5)
                                    if resp.ok:
                                        health_data = resp.json()
                                        status["services"][name] = {
                                            "status": "healthy",
                                            "pid": pid,
                                            "embedding_loaded": health_data.get("embedding_loaded"),
                                            "reranker_loaded": health_data.get("reranker_loaded")
                                        }
                                    else:
                                        status["services"][name] = {
                                            "status": "degraded",
                                            "pid": pid,
                                            "reason": f"health check failed: {resp.status_code}"
                                        }
                                except Exception as e:
                                    status["services"][name] = {
                                        "status": "degraded",
                                        "pid": pid,
                                        "reason": f"health check error: {str(e)}"
                                    }
                        else:
                            # 其他服务仅用 PID 判断
                            status["services"][name] = {
                                "status": "running" if is_running else "stopped",
                                "pid": pid
                            }
    except Exception as e:
        status["services"]["error"] = str(e)

    # 检查 Docker
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        status["docker"][parts[0]] = parts[1]
    except Exception as e:
        status["docker"]["error"] = str(e)

    return json.dumps(status, ensure_ascii=False, indent=2)


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