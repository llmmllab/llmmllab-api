"""
Web search tool using SearxNG provider with LangGraph Command pattern.

This module provides a single, streamlined web search tool that integrates cleanly
with LangGraph workflows using strong typing and efficient state access.

Features:
- Single function-based tool using @tool decorator
- Strong typing with WorkflowState instead of generic Dict
- Efficient user_config access from injected state (no database calls)
- Command pattern for proper state updates
- SearxNG provider with comprehensive engine support

Configuration:
- Default engines: Google, Bing, DuckDuckGo, Startpage for comprehensive coverage
- Structured results API for reliable parsing
- User-specific preferences from WorkflowState.user_config.web_search
- Configurable engines, categories, and search parameters

Usage in LangGraph workflows:
    # Tool is automatically available when registered in tool registry
    # LangGraph handles injection of tool_call_id and WorkflowState

Available Engines (see https://docs.searxng.org/dev/engines/index.html):
- Web: google, bing, duckduckgo, startpage, yahoo, yandex
- Academic: google_scholar, arxiv, crossref, semantic_scholar
- News: google_news, bing_news, yahoo_news, reddit
- Technical: github, stackoverflow, gitlab
- Shopping: google_shopping, bing_shopping, amazon, ebay
- And many more specialized engines
"""

from typing import Annotated, List, Literal, Optional

import httpx

from langchain_core.tools import tool
from config import SEARX_HOST
from utils.logging import llmmllogger
from models import SearchResult, SearchResultContent, WebSearchConfig


# Native SearxNG wrapper — replaces langchain_community SearxSearchWrapper
# to eliminate the langchain-community dependency.
class SearxNGWrapper:
    """Lightweight SearxNG wrapper using httpx (no langchain_community needed)."""

    def __init__(
        self,
        searx_host: str,
        engines: Optional[List[str]] = None,
        k: int = 10,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        categories: Optional[List[str]] = None,
    ):
        self.searx_host = searx_host.rstrip("/")
        self.engines = engines
        self.k = k
        self.params = params or {}
        self.headers = headers or {}
        self.categories = categories or ["general"]

    def results(
        self,
        query: str,
        num_results: Optional[int] = None,
        engines: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> list[dict]:
        """Execute a search and return structured results as dicts."""
        search_params = {
            "q": query,
            "format": "json",
            "categories": ",".join(categories or self.categories),
        }
        if engines:
            search_params["engines"] = ",".join(engines)
        elif self.engines:
            search_params["engines"] = ",".join(self.engines)

        # Merge with default params (language, safesearch, etc.)
        search_params.update({k: v for k, v in self.params.items() if k not in search_params})

        n = num_results or self.k

        try:
            resp = httpx.get(
                f"{self.searx_host}/search",
                params=search_params,
                headers=self.headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])[:n]
            # Normalize to the dict format expected by the caller
            return [
                {
                    "title": r.get("title", ""),
                    "link": r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "Result": "" if r.get("title") else "No good Search Result was found",
                }
                for r in results
            ]
        except Exception as e:
            raise RuntimeError(f"SearxNG search failed: {e}") from e


#   engines:
# type: array
#     description: List of SearxNG search engines to use
#     items:
# type: string
#     default:
#       [
#         "google",
#         "bing",
#         "duckduckgo",
#         "startpage",
#         "github",
#         "arxiv",
#         "wikipedia",
#       ]
#   categories:
# type: array
#     description: SearxNG search categories to include
#     items:
# type: string
#       enum:
#         [
#           "general",
#           "news",
#           "science",
#           "it",
#           "shopping",
#           "images",
#           "videos",
#           "music",
#           "files",
#           "social",
#         ]
#     default: ["general", "news", "science", "it"]


class SearxNG:
    """Wrapper for Searx Search API using WebSearchConfig."""

    def __init__(
        self,
        web_config: WebSearchConfig,
        categories: List[
            Literal[
                "general",
                "news",
                "science",
                "it",
                "shopping",
                "images",
                "videos",
                "music",
                "files",
                "social",
            ]
        ],
    ):
        self.web_config = web_config
        self.searx_host = web_config.searx_host or SEARX_HOST
        self.categories = categories or list[str](web_config.categories)

        # Build search parameters directly from WebSearchConfig
        params = {
            "format": "json",
            "language": web_config.language,
            "safesearch": web_config.safesearch,
            "time_range": web_config.time_range or "",
        }

        headers = {
            "User-Agent": web_config.user_agent or "LLMMLLab-WebSearch/1.0",
        }

        self.wrapper = SearxNGWrapper(
            searx_host=self.searx_host,
            engines=web_config.engines,
            k=web_config.max_results,
            params=params,
            headers=headers,
            categories=self.categories,  # type: ignore
        )
        self.logger = llmmllogger.logger

        self.logger.debug(f"SearxNG initialized with engines: {web_config.engines}")

    async def search(
        self,
        query: str,
        max_results: int,
        categories: List[
            Literal[
                "general",
                "news",
                "science",
                "it",
                "shopping",
                "images",
                "videos",
                "music",
                "files",
                "social",
            ]
        ] = ["general"],
    ) -> SearchResult:
        """Execute search using Searx Search API."""
        results = []
        error = None
        try:
            if not query.strip():
                return SearchResult(
                    is_from_url_in_user_query=False,
                    query=query,
                    contents=results,
                    error="Empty query",
                )
            # Use the results() method for structured data instead of run()
            # This gives us proper metadata and structured results
            structured_results = self.wrapper.results(
                query=query,
                num_results=max_results,
                engines=None,  # Use configured engines
                categories=categories,  # type: ignore
            )

            # Convert structured results to our format
            for i, result in enumerate(structured_results):
                if (
                    "Result" in result
                    and result["Result"] == "No good Search Result was found"
                ):
                    continue

                url = result.get("link", "No URL")
                if url.endswith("robots.txt"):
                    self.logger.debug(f"Skipping robots.txt URL: {url}")
                    continue

                title = result.get("title", "No title")
                content = result.get("snippet", "No content")

                results.append(
                    SearchResultContent(
                        url=url,
                        title=title,
                        content=content,
                        relevance=1.0 - (0.05 * i),
                    )
                )

            return SearchResult(
                is_from_url_in_user_query=False,
                query=query,
                contents=results,
                error=error,
            )

        except Exception as e:
            error = f"Error with Searx search: {e}"
            self.logger.error(error)

            return SearchResult(
                is_from_url_in_user_query=False,
                query=query,
                contents=results,
                error=error,
            )


# Single web search tool using simplified signature for testing
@tool
async def web_search(
    query: Annotated[str, "The search query to execute"],
    num_results: Annotated[Optional[int], "Number of search results to return"] = None,
    categories: Annotated[
        List[
            Literal[
                "general",
                "news",
                "science",
                "it",
                "shopping",
                "images",
                "videos",
                "music",
                "files",
                "social",
            ]
        ],
        "Search categories to include",
    ] = ["general"],
) -> str:
    """
    Search the web for information and automatically add results to workflow state.

    This tool performs comprehensive web searches using multiple search engines
    and returns structured results. Use this tool when you need current information
    from the internet about any topic.

    Args:
        query: The search query to execute
        num_results: Number of search results to return (overrides user config if provided)

    Returns:
        Search results with titles, URLs, content snippets, and relevance scores
    """
    from models.default_configs import (  # pylint: disable=import-outside-toplevel
        DEFAULT_WEB_SEARCH_CONFIG,
    )

    logger = llmmllogger.bind(component="WebSearch")
    try:
        # For testing without ToolRuntime - use default config
        # TODO: Implement proper LangGraph agent context to support ToolRuntime
        web_config = DEFAULT_WEB_SEARCH_CONFIG
        logger.debug(
            "Using default web search config - ToolRuntime temporarily removed for testing"
        )
        if not num_results:
            num_results = DEFAULT_WEB_SEARCH_CONFIG.max_results

        # Use SearxNG provider with WebSearchConfig
        provider = SearxNG(web_config=web_config, categories=categories)
        search_result = await provider.search(query, num_results)
        if search_result and search_result.contents:
            return search_result.model_dump_json()

        # No results found
        if search_result and search_result.error:
            return f"⚠️ Web search error: {search_result.error}"

        return f"🔍 No results found for query: '{query}'"

    except Exception as e:
        error_message = f"❌ Web search failed: {str(e)}"
        logger.error(f"Web search error: {e}", query=query, error=str(e))

        return error_message
