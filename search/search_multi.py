"""
多后端聚合搜索模块 — 单入口 + 用量追踪 + 本地缓存 + 诊断

降级链: Exa (语义搜索) → Tavily (AI优化) → DDGS (DuckDuckGo, 零成本保底)

用法:
    >>> from search.search_multi import search, get_quota_status
    >>> result = search("Python 异步编程", count=5)
    >>> print(result["provider"], result["count"])
    Exa (语义) 2

    >>> status = get_quota_status()  # 查看各 provider 用量
"""

import hashlib
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
CACHE_FILE = Path.home() / "zhiwei-rag" / "data" / "search_cache.json"
DIAG_FILE = Path.home() / "zhiwei-rag" / "data" / "search_diag.json"

DEFAULT_LIMITS = {
    "tavily": 1000,
    "exa": 1000,
}

CACHE_TTL_SECONDS = 300  # 5 分钟
DIAG_MAX_ENTRIES = 50

# ============================================================================
# 配置加载
# ============================================================================


_KEYS_CACHE: dict = {}
_KEYS_LOADED: bool = False


def _load_keys() -> dict:
    """从 global.env 加载 API keys（带缓存）"""
    global _KEYS_LOADED
    if _KEYS_LOADED:
        return _KEYS_CACHE

    env_path = Path.home() / ".secrets" / "global.env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("TAVILY_API_KEY="):
                _KEYS_CACHE["tavily"] = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("EXA_API_KEY="):
                _KEYS_CACHE["exa"] = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("SERPER_API_KEY="):
                _KEYS_CACHE["serper"] = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("BRAVE_API_KEY="):
                _KEYS_CACHE["brave"] = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("SEARXNG_URL="):
                _KEYS_CACHE["searxng"] = line.split("=", 1)[1].strip().strip('"')
    _KEYS_LOADED = True
    return _KEYS_CACHE


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
# 本地结果缓存
# ============================================================================


class SearchCache:
    """本地 JSON 文件缓存，TTL 5 分钟"""

    def __init__(self):
        self._data = self._load()

    def _load(self) -> dict:
        if CACHE_FILE.exists():
            try:
                return json.loads(CACHE_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save(self):
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(self._data, ensure_ascii=False, indent=2))

    @staticmethod
    def _key(query: str, count: int) -> str:
        raw = f"{query.lower().strip()}|{count}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, count: int) -> dict | None:
        k = self._key(query, count)
        entry = self._data.get(k)
        if not entry:
            return None
        if time.time() - entry["ts"] > CACHE_TTL_SECONDS:
            del self._data[k]
            self._save()
            return None
        return entry["result"]

    def put(self, query: str, count: int, result: dict):
        k = self._key(query, count)
        self._data[k] = {"ts": time.time(), "result": result}
        # 清理过期条目
        now = time.time()
        self._data = {
            kk: vv for kk, vv in self._data.items()
            if now - vv["ts"] <= CACHE_TTL_SECONDS
        }
        self._save()


_cache = SearchCache()


# ============================================================================
# 诊断日志
# ============================================================================

_diag_entries: list[dict] = []


def _record_diag(provider: str, status: str, elapsed: float, error: str = "", hit_cache: bool = False):
    global _diag_entries
    entry = {
        "time": datetime.now().isoformat(),
        "provider": provider,
        "status": status,  # success / error / cache_hit
        "elapsed_s": round(elapsed, 3),
        "hit_cache": hit_cache,
    }
    if error:
        entry["error"] = error
    _diag_entries.append(entry)
    _diag_entries = _diag_entries[-DIAG_MAX_ENTRIES:]

    # 持久化
    try:
        DIAG_FILE.parent.mkdir(parents=True, exist_ok=True)
        DIAG_FILE.write_text(json.dumps(_diag_entries, ensure_ascii=False, indent=2))
    except OSError:
        pass


def get_diagnostics() -> list[dict]:
    """返回最近诊断记录"""
    if not _diag_entries and DIAG_FILE.exists():
        try:
            _diag_entries.extend(json.loads(DIAG_FILE.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    return _diag_entries[-10:]


# ============================================================================
# Provider: Exa (语义搜索)
# ============================================================================

def _search_exa(query: str, count: int, api_key: str) -> list[dict]:
    """Exa AI Semantic Search — 20s 超时 + 2 次重试"""
    try:
        import httpx
    except ImportError:
        logger.warning("httpx 未安装，跳过 Exa 搜索")
        return []

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

    last_err = ""
    for attempt in range(3):
        try:
            with httpx.Client(timeout=20) as client:
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
        except Exception as e:
            last_err = str(e)
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
                logger.debug(f"Exa 重试 {attempt + 1}/2: {e}")

    raise RuntimeError(f"Exa 3 次尝试均失败: {last_err}")


# ============================================================================
# Provider: Tavily (AI优化搜索)
# ============================================================================

def _search_tavily(query: str, count: int, api_key: str) -> list[dict]:
    """Tavily Search API (AI-optimized) — 20s 超时"""
    try:
        from tavily import TavilyClient
    except ImportError:
        logger.warning("tavily-python SDK 未安装，跳过 Tavily 搜索")
        return []

    client = TavilyClient(api_key=api_key, timeout=20)
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
    """DuckDuckGo Search (无需 API key) — 带异常保护"""
    from ddgs import DDGS

    results = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=count, region="cn-zh"):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", "")
                })
    except Exception as e:
        raise RuntimeError(f"DDGS 搜索失败: {e}") from e

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
    多后端聚合搜索，自动降级 + 用量追踪 + 本地缓存 + 诊断

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
        或 {"error": "...", "diagnostics": "..."}
    """
    # 1. 检查缓存
    cached = _cache.get(query, count)
    if cached:
        _record_diag("cache", "cache_hit", 0, hit_cache=True)
        logger.info(f"缓存命中: {query[:50]}")
        cached["_cached"] = True
        return cached

    keys = _load_keys()
    tried = []
    errors = []

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
                result = _make_result(label, results, elapsed)
                # 写入缓存
                _cache.put(query, count, result)
                _record_diag(label, "success", elapsed)
                logger.info(f"搜索成功: {label} (耗时 {elapsed:.2f}s, {len(results)} 条)")
                return result
            else:
                logger.debug(f"{label} 返回空结果")

        except Exception as e:
            err_msg = f"{label}: {e}"
            errors.append(err_msg)
            elapsed = time.monotonic() - start if 'start' in dir() else 0
            _record_diag(label, "error", elapsed, error=err_msg)
            logger.debug(f"{label} 失败: {e}")
            continue

    # 所有 provider 均失败，返回诊断信息
    diag_summary = " | ".join(errors) if errors else "无可用 provider"
    return {
        "error": f"所有搜索引擎均失败 (尝试了: {', '.join(tried)})",
        "diagnostics": diag_summary
    }


def get_quota_status() -> dict:
    """返回所有 provider 的用量状态"""
    return _quota.get_status()
