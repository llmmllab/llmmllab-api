# SearxNG Web Search Integration

## Overview

This module provides a native, zero-dependency web search integration using [SearxNG](https://docs.searxng.org/). It replaces the previous `langchain-community` dependency (`SearxSearchWrapper`) with a lightweight custom implementation using `httpx`.

## Architecture

```
web_search (tool)
  └── SearxNG (provider wrapper)
       └── SearxNGWrapper (HTTP client)
            └── httpx → SearxNG /search endpoint
```

### Components

| Class | Purpose |
|-------|---------|
| `SearxNGWrapper` | Low-level HTTP client for SearxNG's `/search` endpoint. Handles request params, response normalization, and error wrapping. |
| `SearxNG` | Mid-level provider that bridges `WebSearchConfig` to `SearxNGWrapper`. Manages per-user config (engines, categories, language, safe search). |
| `web_search` | LangChain `@tool` decorated async function. The entry point for LangGraph workflows. |

## Configuration

All configuration flows through `WebSearchConfig` (defined in `models/`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `searx_host` | `str` | `SEARX_HOST` env var | SearxNG instance URL |
| `engines` | `List[str]` | `None` | Search engines to use (google, bing, duckduckgo, etc.) |
| `max_results` | `int` | `10` | Maximum results per query |
| `language` | `str` | `""` | Search language code |
| `safesearch` | `int` | `0` | Safe search level (0=off, 1=moderate, 2=strict) |
| `time_range` | `str` | `""` | Time filter (day, week, month, year) |
| `user_agent` | `str` | `"LLMMLLab-WebSearch/1.0"` | HTTP User-Agent header |
| `categories` | `List[str]` | `["general"]` | Search categories |

## Available SearxNG Engines

See the [SearxNG engine docs](https://docs.searxng.org/dev/engines/index.html) for the full list. Common engines:

- **Web**: `google`, `bing`, `duckduckgo`, `startpage`, `yahoo`, `yandex`
- **Academic**: `google_scholar`, `arxiv`, `crossref`, `semantic_scholar`
- **News**: `google_news`, `bing_news`, `yahoo_news`, `reddit`
- **Technical**: `github`, `stackoverflow`, `gitlab`

## Testing

Unit tests are in `test/unit/tools/test_searxng_wrapper.py`:

```bash
uv run pytest test/unit/tools/test_searxng_wrapper.py -v
```

Test coverage includes:
- `SearxNGWrapper` construction and parameter defaults
- `SearxNGWrapper.results()` — success, error, empty results, result limiting
- HTTP error handling (500, connection errors, timeouts)
- `SearxNG` construction from `WebSearchConfig`
- `SearxNG.search()` — success, empty query, error propagation, relevance scoring
- Category and engine passthrough

## Migration Notes (from langchain-community)

The previous implementation used `langchain_community.utilities.SearxSearchWrapper`. The new `SearxNGWrapper` provides the same API surface:

| Old (langchain-community) | New (native) |
|---------------------------|--------------|
| `SearxSearchWrapper(searx_host=...)` | `SearxNGWrapper(searx_host=...)` |
| `wrapper.run(query)` | `wrapper.results(query)` |
| Returns `str` | Returns `list[dict]` with `title`, `link`, `snippet`, `Result` |

The `SearxNG` class wraps `SearxNGWrapper` and provides the same `search()` async interface as before, returning `SearchResult` objects.
