"""
知微 MCP Server
提供 RAG 检索和系统状态查询工具

使用 FastMCP 实现 (官方 Python SDK)
参考：https://github.com/modelcontextprotocol/python-sdk
"""

import subprocess
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
    import requests
    import json

    try:
        response = requests.post(
            "http://127.0.0.1:8765/search",
            json={"query": query, "top_k": top_k},
            timeout=30
        )

        if response.ok:
            data = response.json()
            results = data.get("results", [])

            if not results:
                return json.dumps({"status": "success", "message": "未找到相关结果"})

            # 格式化结果
            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append({
                    "rank": i,
                    "text": r.get("text", "")[:500],
                    "score": r.get("score", 0),
                    "source": r.get("source", "unknown")
                })

            return json.dumps({
                "query": query,
                "count": len(formatted),
                "results": formatted
            }, ensure_ascii=False, indent=2)
        else:
            return f"错误: rag-api 返回 HTTP {response.status_code}"

    except requests.exceptions.ConnectionError:
        return "错误: rag-api 服务未运行 (127.0.0.1:8765)"
    except Exception as e:
        return f"错误: {str(e)}"


@mcp.tool()
def get_system_health() -> str:
    """获取知微系统健康状态，包括服务和 Docker 容器状态"""
    import json

    status = {
        "services": {},
        "docker": {},
        "rag_api": {}
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
                    pid, status_code, name = parts[0], parts[1], parts[2]
                    if name in services:
                        status["services"][name] = {
                            "status": "running" if status_code == "0" else "error",
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

    # 检查 rag-api
    try:
        import requests
        response = requests.get("http://127.0.0.1:8765/health", timeout=5)
        if response.ok:
            status["rag_api"] = response.json()
    except:
        status["rag_api"]["status"] = "unreachable"

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


if __name__ == "__main__":
    mcp.run()