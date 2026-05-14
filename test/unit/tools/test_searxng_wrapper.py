"""Unit tests for SearxNGWrapper and SearxNG (native SearxNG integration).

These tests validate the native httpx-based SearxNG wrapper that replaced
the langchain-community dependency in PR #59 / issue #25.

Covers:
- SearxNGWrapper construction and parameter defaults
- SearxNGWrapper.results() success, error, and edge cases
- SearxNG construction from WebSearchConfig
- SearxNG.search() success, empty query, error handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

import importlib.util as _importlib_util
import os as _os
import sys as _sys

# Avoid circular imports through tools.static.__init__ by loading the module file directly
# __file__ is test/unit/tools/test_searxng_wrapper.py → go up 4 levels to repo root
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
_web_search_path = _os.path.join(_repo_root, 'tools', 'static', 'web_search_tool.py')
_web_spec = _importlib_util.spec_from_file_location('_web_search_tool', _web_search_path)
_web_mod = _importlib_util.module_from_spec(_web_spec)
# Pre-register dependencies the module needs
import httpx as _httpx
_sys.modules['httpx'] = _httpx
_sys.modules['_web_search_tool'] = _web_mod
_web_spec.loader.exec_module(_web_mod)

SearxNGWrapper = _web_mod.SearxNGWrapper
SearxNG = _web_mod.SearxNG

from models import WebSearchConfig, SearchResult, SearchResultContent


# ── SearxNGWrapper tests ──────────────────────────────────────────────


class TestSearxNGWrapperInit:
    """Test SearxNGWrapper constructor and defaults."""

    def test_default_parameters(self):
        wrapper = SearxNGWrapper(searx_host="http://localhost:8080/")
        assert wrapper.searx_host == "http://localhost:8080"
        assert wrapper.engines is None
        assert wrapper.k == 10
        assert wrapper.params == {}
        assert wrapper.headers == {}
        assert wrapper.categories == ["general"]

    def test_custom_parameters(self):
        wrapper = SearxNGWrapper(
            searx_host="http://searx.example.com",
            engines=["google", "bing"],
            k=5,
            params={"language": "en", "safesearch": 1},
            headers={"User-Agent": "TestBot/1.0"},
            categories=["news", "science"],
        )
        assert wrapper.searx_host == "http://searx.example.com"
        assert wrapper.engines == ["google", "bing"]
        assert wrapper.k == 5
        assert wrapper.params == {"language": "en", "safesearch": 1}
        assert wrapper.headers == {"User-Agent": "TestBot/1.0"}
        assert wrapper.categories == ["news", "science"]

    def test_trailing_slash_stripped(self):
        wrapper = SearxNGWrapper(searx_host="http://localhost:8080///")
        assert wrapper.searx_host == "http://localhost:8080"


class TestSearxNGWrapperResults:
    """Test SearxNGWrapper.results() HTTP call and response parsing."""

    @pytest.fixture
    def wrapper(self):
        return SearxNGWrapper(
            searx_host="http://searx.test",
            engines=["google"],
            k=5,
            params={"language": "en"},
            headers={"User-Agent": "TestBot"},
            categories=["general"],
        )

    def _mock_response(self, json_data, status_code=200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data
        mock_resp.raise_for_status = MagicMock()
        if status_code >= 400:
            mock_resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
        return mock_resp

    @patch("httpx.get")
    def test_successful_search(self, mock_get, wrapper):
        mock_get.return_value = self._mock_response({
            "results": [
                {"title": "Result 1", "url": "http://example.com/1", "content": "Snippet 1"},
                {"title": "Result 2", "url": "http://example.com/2", "content": "Snippet 2"},
            ]
        })

        results = wrapper.results("test query")

        assert len(results) == 2
        assert results[0]["title"] == "Result 1"
        assert results[0]["link"] == "http://example.com/1"
        assert results[0]["snippet"] == "Snippet 1"
        assert results[0]["Result"] == ""

        # Verify the HTTP call
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs[0][0] == "http://searx.test/search"
        params = call_kwargs[1]["params"]
        assert params["q"] == "test query"
        assert params["format"] == "json"
        assert params["categories"] == "general"
        assert params["engines"] == "google"
        assert params["language"] == "en"

    @patch("httpx.get")
    def test_num_results_limits_output(self, mock_get, wrapper):
        mock_get.return_value = self._mock_response({
            "results": [
                {"title": f"R{i}", "url": f"http://x.com/{i}", "content": f"S{i}"}
                for i in range(10)
            ]
        })

        results = wrapper.results("test", num_results=3)
        assert len(results) == 3

    @patch("httpx.get")
    def test_default_k_limits_output(self, mock_get, wrapper):
        mock_get.return_value = self._mock_response({
            "results": [
                {"title": f"R{i}", "url": f"http://x.com/{i}", "content": f"S{i}"}
                for i in range(20)
            ]
        })

        results = wrapper.results("test")
        # wrapper.k=5, so max 5 results
        assert len(results) == 5

    @patch("httpx.get")
    def test_empty_results(self, mock_get, wrapper):
        mock_get.return_value = self._mock_response({"results": []})
        results = wrapper.results("no results query")
        assert results == []

    @patch("httpx.get")
    def test_result_without_title(self, mock_get, wrapper):
        mock_get.return_value = self._mock_response({
            "results": [
                {"title": "", "url": "http://example.com", "content": "Some content"},
            ]
        })
        results = wrapper.results("test")
        assert len(results) == 1
        assert results[0]["Result"] == "No good Search Result was found"

    @patch("httpx.get")
    def test_engines_override(self, mock_get, wrapper):
        mock_get.return_value = self._mock_response({"results": []})
        wrapper.results("test", engines=["duckduckgo", "bing"])

        params = mock_get.call_args[1]["params"]
        assert params["engines"] == "duckduckgo,bing"

    @patch("httpx.get")
    def test_categories_override(self, mock_get, wrapper):
        mock_get.return_value = self._mock_response({"results": []})
        wrapper.results("test", categories=["news", "it"])

        params = mock_get.call_args[1]["params"]
        assert params["categories"] == "news,it"

    @patch("httpx.get")
    def test_http_error_raises_runtime_error(self, mock_get, wrapper):
        mock_get.return_value = self._mock_response({}, status_code=500)
        with pytest.raises(RuntimeError, match="SearxNG search failed"):
            wrapper.results("test")

    @patch("httpx.get")
    def test_connection_error_raises_runtime_error(self, mock_get, wrapper):
        mock_get.side_effect = ConnectionError("refused")
        with pytest.raises(RuntimeError, match="SearxNG search failed"):
            wrapper.results("test")

    @patch("httpx.get")
    def test_timeout_raises_runtime_error(self, mock_get, wrapper):
        mock_get.side_effect = TimeoutError("timed out")
        with pytest.raises(RuntimeError, match="SearxNG search failed"):
            wrapper.results("test")

    @patch("httpx.get")
    def test_request_includes_headers(self, mock_get, wrapper):
        mock_get.return_value = self._mock_response({"results": []})
        wrapper.results("test")

        headers = mock_get.call_args[1]["headers"]
        assert headers["User-Agent"] == "TestBot"

    @patch("httpx.get")
    def test_request_has_timeout(self, mock_get, wrapper):
        mock_get.return_value = self._mock_response({"results": []})
        wrapper.results("test")

        assert mock_get.call_args[1]["timeout"] == 10.0

    @patch("httpx.get")
    def test_params_merged_without_overwriting_core(self, mock_get, wrapper):
        """Default params (language, safesearch) should not overwrite q/format/categories."""
        mock_get.return_value = self._mock_response({"results": []})
        wrapper.results("test")

        params = mock_get.call_args[1]["params"]
        # Core params must not be overwritten by default params
        assert params["q"] == "test"
        assert params["format"] == "json"


# ── SearxNG tests ─────────────────────────────────────────────────────


class TestSearxNGInit:
    """Test SearxNG construction from WebSearchConfig."""

    def test_init_with_defaults(self):
        config = WebSearchConfig()
        searxng = SearxNG(web_config=config, categories=["general"])
        assert searxng.categories == ["general"]
        assert searxng.wrapper is not None
        assert searxng.wrapper.k == config.max_results

    def test_init_with_custom_config(self):
        config = WebSearchConfig(
            searx_host="http://custom-searx.test",
            engines=["google", "bing"],
            max_results=15,
            language="fr",
            safesearch=2,
            user_agent="CustomBot/2.0",
        )
        searxng = SearxNG(web_config=config, categories=["news"])
        assert searxng.searx_host == "http://custom-searx.test"
        assert searxng.categories == ["news"]
        assert searxng.wrapper.engines == ["google", "bing"]
        assert searxng.wrapper.k == 15
        assert searxng.wrapper.params["language"] == "fr"
        assert searxng.wrapper.params["safesearch"] == 2
        assert searxng.wrapper.headers["User-Agent"] == "CustomBot/2.0"


class TestSearxNGSearch:
    """Test SearxNG.search() async method."""

    @pytest.fixture
    def searxng(self):
        config = WebSearchConfig(searx_host="http://searx.test")
        return SearxNG(web_config=config, categories=["general"])

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self, searxng):
        result = await searxng.search("", max_results=5)
        assert isinstance(result, SearchResult)
        assert result.error == "Empty query"
        assert result.contents == []

    @pytest.mark.asyncio
    async def test_whitespace_only_query_returns_error(self, searxng):
        result = await searxng.search("   ", max_results=5)
        assert isinstance(result, SearchResult)
        assert result.error == "Empty query"

    @pytest.mark.asyncio
    async def test_successful_search(self, searxng):
        """Happy path: search returns results."""
        with patch.object(searxng.wrapper, "results") as mock_results:
            mock_results.return_value = [
                {
                    "title": "Python Docs",
                    "link": "https://docs.python.org",
                    "snippet": "Official Python documentation",
                    "Result": "",
                },
                {
                    "title": "Real Python",
                    "link": "https://realpython.com",
                    "snippet": "Python tutorials",
                    "Result": "",
                },
            ]

            result = await searxng.search("python tutorial", max_results=5)

            assert isinstance(result, SearchResult)
            assert result.query == "python tutorial"
            assert len(result.contents) == 2
            assert result.contents[0].title == "Python Docs"
            assert result.contents[0].url == "https://docs.python.org"
            assert result.contents[0].content == "Official Python documentation"
            assert result.contents[0].relevance == 1.0
            assert result.contents[1].relevance == 0.95
            assert result.error is None

    @pytest.mark.asyncio
    async def test_skips_no_good_result(self, searxng):
        """Results with 'No good Search Result was found' are skipped."""
        with patch.object(searxng.wrapper, "results") as mock_results:
            mock_results.return_value = [
                {
                    "title": "",
                    "link": "http://example.com",
                    "snippet": "",
                    "Result": "No good Search Result was found",
                },
                {
                    "title": "Good Result",
                    "link": "http://good.com",
                    "snippet": "This is good",
                    "Result": "",
                },
            ]

            result = await searxng.search("test", max_results=5)
            assert len(result.contents) == 1
            assert result.contents[0].title == "Good Result"

    @pytest.mark.asyncio
    async def test_skips_robots_txt(self, searxng):
        """URLs ending in robots.txt are skipped."""
        with patch.object(searxng.wrapper, "results") as mock_results:
            mock_results.return_value = [
                {
                    "title": "Robots",
                    "link": "http://example.com/robots.txt",
                    "snippet": "User-agent: *",
                    "Result": "",
                },
                {
                    "title": "Real Page",
                    "link": "http://example.com/page",
                    "snippet": "Content",
                    "Result": "",
                },
            ]

            result = await searxng.search("test", max_results=5)
            assert len(result.contents) == 1
            assert result.contents[0].url == "http://example.com/page"

    @pytest.mark.asyncio
    async def test_wrapper_error_returns_search_result_with_error(self, searxng):
        """When wrapper raises, search returns SearchResult with error message."""
        with patch.object(searxng.wrapper, "results") as mock_results:
            mock_results.side_effect = RuntimeError("SearxNG search failed: connection refused")

            result = await searxng.search("test", max_results=5)

            assert isinstance(result, SearchResult)
            assert result.error is not None
            assert "Error with Searx search" in result.error
            assert result.contents == []

    @pytest.mark.asyncio
    async def test_relevance_decreases_with_rank(self, searxng):
        """Relevance scores decrease by 0.05 per position."""
        with patch.object(searxng.wrapper, "results") as mock_results:
            mock_results.return_value = [
                {
                    "title": f"Result {i}",
                    "link": f"http://x.com/{i}",
                    "snippet": f"Snippet {i}",
                    "Result": "",
                }
                for i in range(5)
            ]

            result = await searxng.search("test", max_results=5)
            assert result.contents[0].relevance == 1.0
            assert result.contents[1].relevance == 0.95
            assert result.contents[2].relevance == 0.90
            assert result.contents[3].relevance == 0.85
            assert result.contents[4].relevance == 0.80

    @pytest.mark.asyncio
    async def test_passes_categories_to_wrapper(self, searxng):
        """Categories parameter is passed through to wrapper.results()."""
        with patch.object(searxng.wrapper, "results") as mock_results:
            mock_results.return_value = []

            await searxng.search("test", max_results=5, categories=["news", "science"])

            mock_results.assert_called_once()
            call_kwargs = mock_results.call_args
            assert call_kwargs[1]["categories"] == ["news", "science"]
