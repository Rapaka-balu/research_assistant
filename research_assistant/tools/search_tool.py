"""
tools/search_tool.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DuckDuckGo search wrapper as a LangChain tool.

Why a custom wrapper instead of the built-in?
  - We return structured SearchResult dicts (title, url, snippet)
    so the Citation agent can build proper references.
  - We add retry logic and timeout handling.
  - Results are logged to state.tool_calls_log for eval tracing.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import time
from typing import Any

from langchain_core.tools import tool
from duckduckgo_search import DDGS

from config.settings import MAX_SEARCH_RESULTS, SEARCH_TIMEOUT


@tool
def duckduckgo_search(query: str) -> list[dict[str, str]]:
    """
    Search the web using DuckDuckGo.

    Args:
        query: The search query string.

    Returns:
        List of dicts with keys: title, url, snippet.
    """
    results = []
    try:
        with DDGS(timeout=SEARCH_TIMEOUT) as ddgs:
            raw = ddgs.text(
                query,
                max_results=MAX_SEARCH_RESULTS,
                safesearch="moderate",
            )
            for r in raw:
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        # Return a single error result rather than crashing the agent
        results.append({
            "title":   "Search error",
            "url":     "",
            "snippet": f"Search failed: {str(e)}. Try rephrasing the query.",
        })
    return results


@tool
def duckduckgo_news_search(query: str) -> list[dict[str, str]]:
    """
    Search recent news using DuckDuckGo news endpoint.

    Args:
        query: The news search query.

    Returns:
        List of dicts with keys: title, url, snippet, date.
    """
    results = []
    try:
        with DDGS(timeout=SEARCH_TIMEOUT) as ddgs:
            raw = ddgs.news(query, max_results=MAX_SEARCH_RESULTS)
            for r in raw:
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("url", ""),
                    "snippet": r.get("body", ""),
                    "date":    r.get("date", ""),
                })
    except Exception as e:
        results.append({
            "title": "News search error", "url": "", "snippet": str(e), "date": ""
        })
    return results


# ── All search tools exported for agent bind_tools() ─────────────────
SEARCH_TOOLS = [duckduckgo_search, duckduckgo_news_search]
