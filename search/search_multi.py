"""
多后端聚合搜索模块 — 单入口 + 用量追踪

降级链: Exa (语义搜索) → Tavily (AI优化) → DDGS (DuckDuckGo, 零成本保底)

用法:
    >>> from search.search_multi import search, get_quota_status
    >>> result = search("Python 异步编程", count=5)
    >>> print(result["provider"], result["count"])
    Exa (语义) 2

    >>> status = get_quota_status()  # 查看各 provider 用量
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("zhiwei-rag.search_multi")

# ============================================================================
# 常量
# ============================================================================

QUOTA_FILE = Path.home() / "zhiwei-rag" / "data" / "search_quota.json"

DEFAULT_LIMITS = {
    "tavily": 1000,
    "exa": 1000,
}

# ============================================================================
# 配置加载
# ============================================================================


def _load_keys() -> dict:
    """从 global.env 加载 API keys"""
    keys: dict = {}
    env_path = Path.home() / ".secrets" / "global.env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("TAVILY_API_KEY="):
                keys["tavily"] = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("EXA_API_KEY="):
                keys["exa"] = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("SERPER_API_KEY="):
                keys["serper"] = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("BRAVE_API_KEY="):
                keys["brave"] = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("SEARXNG_URL="):
                keys["searxng"] = line.split("=", 1)[1].strip().strip('"')
    return keys


# ============================================================================
# 用量追踪
# ============================================================================


class QuotaTracker:
    """月度 API 用量追踪"""

    def __init__(self):
        self._data = self._load()

    @property
    def data_dir(self) -> Path:
        return QUOTA_FILE.parent

    def _load(self) -> dict:
        if QUOTA_FILE.exists():
            try:
                data = json.loads(QUOTA_FILE.read_text())
                if "monthly_limits" not in data:
                    data["monthly_limits"] = DEFAULT_LIMITS.copy()
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"monthly_limits": DEFAULT_LIMITS.copy()}

    def _save(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        QUOTA_FILE.write_text(json.dumps(self._data, ensure_ascii=False, indent=2))

    @property
    def _month_key(self) -> str:
        return datetime.now().strftime("%Y-%m")

    def get_usage(self, provider: str) -> int:
        month = self._month_key
        return self._data.get(month, {}).get(provider, 0)

    def get_limit(self, provider: str) -> int:
        return self._data.get("monthly_limits", {}).get(provider, 0)

    def is_within_quota(self, provider: str) -> bool:
        if provider not in DEFAULT_LIMITS:
            return True  # DDGS 无限
        return self.get_usage(provider) < self.get_limit(provider)

    def record_usage(self, provider: str):
        month = self._month_key
        if month not in self._data:
            self._data[month] = {}
        self._data[month][provider] = self._data[month].get(provider, 0) + 1
        self._save()

    def get_status(self) -> dict:
        """返回所有 provider 的用量状态"""
        limits = self._data.get("monthly_limits", DEFAULT_LIMITS)
        month = self._month_key
        usage = self._data.get(month, {})

        status = {}
        for provider, limit in limits.items():
            used = usage.get(provider, 0)
            status[provider] = {
                "used": used,
                "limit": limit,
                "remaining": max(0, limit - used),
                "available": used < limit
            }
        status["ddgs"] = {"used": usage.get("ddgs", 0), "limit": "unlimited", "available": True}
        return status


# 全局实例
_quota = QuotaTracker()


# ============================================================================
# 统一返回格式
# ============================================================================

def _make_result(provider: str, results: list[dict], elapsed: float) -> dict:
    return {
        "provider": provider,
        "count": len(results),
        "results": results,
        "elapsed_seconds": round(elapsed, 2)
    }


# ============================================================================
# Provider: Exa (语义搜索)
# ============================================================================

def _search_exa(query: str, count: int, api_key: str) -> list[dict]:
    """Exa AI Semantic Search"""
    import httpx

    url = "https://api.exa.ai/search"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }
    payload = {
        "query": query,
        "numResults": count,
        "type": "neural",
        "useAutoprompt": True
    }

    with httpx.Client(timeout=15) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        resp_data = resp.json()

    results = []
    for item in resp_data.get("results", [])[:count]:
        snippet = ""
        highlights = item.get("highlights", [])
        if highlights:
            snippet = highlights[0].get("text", "")
        if not snippet:
            snippet = resp_data.get("autopromptString", "")
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": snippet[:300]
        })

    return results


# ============================================================================
# Provider: Tavily (AI优化搜索)
# ============================================================================

def _search_tavily(query: str, count: int, api_key: str) -> list[dict]:
    """Tavily Search API (AI-optimized) — 使用官方 SDK"""
    try:
        from tavily import TavilyClient
    except ImportError:
        logger.warning("tavily-python SDK 未安装，跳过 Tavily 搜索")
        return []

    client = TavilyClient(api_key=api_key)
    resp = client.search(query, max_results=count)

    results = []
    for item in resp.get("results", [])[:count]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", "")
        })

    return results


# ============================================================================
# Provider: DDGS (DuckDuckGo, 零成本保底)
# ============================================================================

def _search_ddgs(query: str, count: int, api_key: str = "") -> list[dict]:
    """DuckDuckGo Search (无需 API key)"""
    from ddgs import DDGS

    results = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=count, region="cn-zh"):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("href", ""),
                "snippet": item.get("body", "")
            })

    return results


# ============================================================================
# 聚合入口
# ============================================================================

# Provider 优先级链: Exa → Tavily → DDGS
# 有 key 且未超限的优先，DDGS 始终兜底
# config_key 是 _load_keys 返回的字典键名（小写）
PROVIDER_CHAIN = [
    ("exa", _search_exa, "exa", "Exa (语义)"),
    ("tavily", _search_tavily, "tavily", "Tavily"),
    ("ddgs", _search_ddgs, None, "DuckDuckGo (保底)"),
]


def search(query: str, count: int = 5) -> dict:
    """
    多后端聚合搜索，自动降级 + 用量追踪

    Args:
        query: 搜索查询
        count: 返回结果数量

    Returns:
        {
            "provider": "xxx",
            "count": N,
            "results": [{"title": "...", "url": "...", "snippet": "..."}],
            "elapsed_seconds": 1.2
        }
        或 {"error": "..."}
    """
    keys = _load_keys()
    tried = []

    for key_name, search_fn, config_key, label in PROVIDER_CHAIN:
        # 无需 API key 的 provider 不受配额限制
        if config_key is not None and not _quota.is_within_quota(key_name):
            logger.warning(f"{label} 月度配额已用尽 ({_quota.get_usage(key_name)}/{_quota.get_limit(key_name)})")
            continue

        # 检查 API key 是否存在
        if config_key:
            if config_key not in keys:
                continue
            api_value = keys[config_key]
        else:
            api_value = ""

        tried.append(label)

        try:
            start = time.monotonic()
            results = search_fn(query, count, api_value)
            elapsed = time.monotonic() - start

            if results and len(results) > 0:
                _quota.record_usage(key_name)
                logger.info(f"搜索成功: {label} (耗时 {elapsed:.2f}s, {len(results)} 条)")
                return _make_result(label, results, elapsed)
            else:
                logger.debug(f"{label} 返回空结果")

        except Exception as e:
            logger.debug(f"{label} 失败: {e}")
            continue

    return {"error": f"所有搜索引擎均失败 (尝试了: {', '.join(tried)})"}


def get_quota_status() -> dict:
    """返回所有 provider 的用量状态"""
    return _quota.get_status()
