"""Tests for search.search_multi module."""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def isolated_search_module(tmp_path, monkeypatch):
    """Re-import search_multi with isolated paths for each test."""
    # Create isolated .secrets
    secrets_dir = tmp_path / ".secrets"
    secrets_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Patch QUOTA_FILE and DEFAULT_LIMITS before import
    quota_file = tmp_path / "zhiwei-rag" / "data" / "search_quota.json"
    quota_file.parent.mkdir(parents=True)

    # Remove cached module so it re-imports with new paths
    mods_to_clear = [k for k in sys.modules if "search_multi" in k]
    for mod in mods_to_clear:
        del sys.modules[mod]

    # We can't patch module-level constants at import, so we use direct
    # monkeypatch on Path.home and let the module pick up the new paths.
    yield tmp_path


@pytest.fixture
def env_with_keys(isolated_search_module):
    """Create .secrets/global.env with API keys."""
    tmp_path = isolated_search_module
    (tmp_path / ".secrets" / "global.env").write_text(
        'TAVILY_API_KEY="tvly-test-123"\n'
        'EXA_API_KEY="exa-test-456"\n'
    )
    return tmp_path


@pytest.fixture
def env_no_keys(isolated_search_module):
    """Create .secrets/global.env with no search keys."""
    tmp_path = isolated_search_module
    (tmp_path / ".secrets" / "global.env").write_text("FOO=bar\n")
    return tmp_path


# ============================================================================
# _load_keys Tests
# ============================================================================


class TestLoadKeys:
    def test_loads_known_keys(self, env_with_keys):
        from search.search_multi import _load_keys

        keys = _load_keys()
        assert keys["tavily"] == "tvly-test-123"
        assert keys["exa"] == "exa-test-456"

    def test_strips_quotes(self, env_with_keys):
        from search.search_multi import _load_keys

        keys = _load_keys()
        # Both quoted and unquoted should be stripped
        assert '"' not in keys.get("tavily", "")

    def test_returns_empty_when_no_keys(self, env_no_keys):
        from search.search_multi import _load_keys

        keys = _load_keys()
        assert keys == {}

    def test_returns_empty_when_env_missing(self, isolated_search_module):
        from search.search_multi import _load_keys

        keys = _load_keys()
        assert keys == {}


# ============================================================================
# QuotaTracker Tests
# ============================================================================


class TestQuotaTracker:
    def _new_tracker(self, tmp_path):
        """Create a QuotaTracker with isolated quota file."""
        from search.search_multi import QuotaTracker
        import search.search_multi as sm

        # Patch QUOTA_FILE to temp location
        original = sm.QUOTA_FILE
        sm.QUOTA_FILE = tmp_path / "quota.json"
        tracker = QuotaTracker()
        sm.QUOTA_FILE = original
        return tracker

    def test_initial_state(self, isolated_search_module):
        tracker = self._new_tracker(isolated_search_module)
        assert tracker.is_within_quota("tavily") is True
        assert tracker.is_within_quota("exa") is True
        assert tracker.get_usage("tavily") == 0

    def test_record_and_check(self, isolated_search_module):
        tracker = self._new_tracker(isolated_search_module)
        assert tracker.is_within_quota("tavily") is True
        tracker.record_usage("tavily")
        assert tracker.get_usage("tavily") == 1
        assert tracker.is_within_quota("tavily") is True

    def test_exceeded_limit(self, isolated_search_module):
        tracker = self._new_tracker(isolated_search_module)
        month = tracker._month_key
        tracker._data[month] = {"tavily": 1000}
        tracker._save()

        assert tracker.is_within_quota("tavily") is False

    def test_unknown_provider_unlimited(self, isolated_search_module):
        tracker = self._new_tracker(isolated_search_module)
        assert tracker.is_within_quota("ddgs") is True

    def test_get_status(self, isolated_search_module):
        tracker = self._new_tracker(isolated_search_module)
        tracker.record_usage("tavily")
        status = tracker.get_status()

        assert "tavily" in status
        assert status["tavily"]["used"] == 1
        assert status["tavily"]["limit"] == 1000
        assert status["tavily"]["remaining"] == 999
        assert status["tavily"]["available"] is True

        assert "ddgs" in status
        assert status["ddgs"]["limit"] == "unlimited"

    def test_data_persistence(self, isolated_search_module):
        quota_file = isolated_search_module / "quota.json"

        import search.search_multi as sm
        from search.search_multi import QuotaTracker

        original = sm.QUOTA_FILE
        sm.QUOTA_FILE = quota_file
        tracker = QuotaTracker()
        tracker.record_usage("tavily")
        sm.QUOTA_FILE = original

        # Reload from the temp file
        assert quota_file.exists()
        data = json.loads(quota_file.read_text())
        assert data[tracker._month_key]["tavily"] == 1

    def test_missing_monthly_limits_repaired(self, isolated_search_module):
        """If quota file lacks monthly_limits, _load() repairs it."""
        quota_file = isolated_search_module / "quota.json"
        quota_file.write_text('{"2026-01": {"tavily": 5}}')

        from search.search_multi import QuotaTracker
        import search.search_multi as sm

        original = sm.QUOTA_FILE
        sm.QUOTA_FILE = quota_file
        tracker = QuotaTracker()
        sm.QUOTA_FILE = original

        assert tracker.get_limit("tavily") == 1000
        assert tracker._data["monthly_limits"]["tavily"] == 1000


# ============================================================================
# search() Tests
# ============================================================================


class TestSearch:
    def test_first_provider_succeeds(self, env_with_keys):
        """First provider with key and quota returns results immediately."""
        from search.search_multi import search as search_fn

        mock_fn = MagicMock(return_value=[
            {"title": "Test", "url": "https://test.com", "snippet": "Test snippet"}
        ])
        with patch("search.search_multi.PROVIDER_CHAIN", [
            ("exa", mock_fn, "exa", "Exa"),
            ("ddgs", MagicMock(return_value=[]), None, "DDGS"),
        ]), patch("search.search_multi._quota") as mock_quota:
            mock_quota.is_within_quota.return_value = True

            result = search_fn("test query", count=1)

            assert result["provider"] == "Exa"
            assert result["count"] == 1
            assert "elapsed_seconds" in result
            mock_fn.assert_called_once()
            mock_quota.record_usage.assert_called_once_with("exa")

    def test_fallback_on_error(self, env_with_keys):
        """When first provider fails, falls through to next."""
        from search.search_multi import search as search_fn

        def raise_error(*args):
            raise ConnectionError("API down")

        def succeed(*args):
            return [{"title": "DDG", "url": "https://ddg.com", "snippet": "DDG result"}]

        with patch("search.search_multi.PROVIDER_CHAIN", [
            ("exa", raise_error, "exa", "Exa"),
            ("ddgs", succeed, None, "DDGS"),
        ]), patch("search.search_multi._quota") as mock_quota:
            mock_quota.is_within_quota.return_value = True

            result = search_fn("test query", count=1)

            assert result["count"] == 1
            assert result["provider"] == "DDGS"

    def test_all_fail_returns_error(self, env_with_keys):
        """All providers failing returns error dict."""
        from search.search_multi import search as search_fn

        def fail(*args):
            raise Exception("fail")

        with patch("search.search_multi.PROVIDER_CHAIN", [
            ("exa", fail, "exa", "Exa"),
            ("ddgs", fail, None, "DDGS"),
        ]), patch("search.search_multi._quota") as mock_quota:
            mock_quota.is_within_quota.return_value = True

            result = search_fn("test query", count=1)

            assert "error" in result
            assert "所有搜索引擎均失败" in result["error"]

    def test_skips_when_quota_exceeded(self, env_with_keys):
        """Provider with exceeded quota is skipped."""
        from search.search_multi import search as search_fn

        mock_fn = MagicMock(return_value=[
            {"title": "DDG", "url": "https://ddg.com", "snippet": "DDG result"}
        ])
        with patch("search.search_multi.PROVIDER_CHAIN", [
            ("exa", MagicMock(side_effect=Exception("fail")), "exa", "Exa"),
            ("ddgs", mock_fn, None, "DDGS"),
        ]), patch("search.search_multi._quota") as mock_quota:
            mock_quota.is_within_quota.side_effect = lambda p: p != "exa"

            result = search_fn("test query", count=1)

            assert result["count"] == 1
            mock_fn.assert_called_once()

    def test_skips_when_key_missing(self, env_no_keys):
        """Provider without API key in env is skipped."""
        from search.search_multi import search as search_fn

        mock_fn = MagicMock(return_value=[
            {"title": "DDG", "url": "https://ddg.com", "snippet": "DDG result"}
        ])
        with patch("search.search_multi.PROVIDER_CHAIN", [
            ("exa", MagicMock(side_effect=Exception("should not reach")), "exa", "Exa"),
            ("ddgs", mock_fn, None, "DDGS"),
        ]), patch("search.search_multi._quota") as mock_quota:
            mock_quota.is_within_quota.return_value = True

            result = search_fn("test query", count=1)

            assert result["count"] == 1


# ============================================================================
# _make_result Tests
# ============================================================================


class TestMakeResult:
    def test_result_format(self):
        from search.search_multi import _make_result

        result = _make_result("TestProvider", [{"title": "T"}], 1.234)

        assert result["provider"] == "TestProvider"
        assert result["count"] == 1
        assert result["elapsed_seconds"] == 1.23
        assert len(result["results"]) == 1
