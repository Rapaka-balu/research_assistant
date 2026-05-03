"""
agents/search_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Search Agent — uses DuckDuckGo tools directly.

Flow:
  1. LLM receives query and generates focused search sub-queries (JSON)
  2. We execute each query via DuckDuckGo directly
  3. Collect and return all results to state

This avoids Groq tool-calling format issues by having the LLM
produce structured JSON queries instead of tool_call objects.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import json
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from state import ResearchState, SearchResult
from tools.search_tool import duckduckgo_search, duckduckgo_news_search
from config.settings import GROQ_API_KEY, GROQ_MODEL

_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)

SEARCH_SYSTEM = """You are a research search specialist. Given a user query, generate 2-3 focused search queries to find comprehensive information.

Reply ONLY with a JSON object in this exact format:
{{"queries": ["search query 1", "search query 2", "search query 3"]}}

Rules:
  - Generate 2-3 specific, focused search queries
  - Break complex topics into sub-queries
  - Use different angles to get comprehensive coverage
  - Reply ONLY with the JSON, no explanation"""


def search_agent_node(state: ResearchState) -> dict:
    """
    Search agent node.
    Has the LLM generate search queries, then executes them directly.
    Returns a partial state update with search results.
    """
    query = state["query"]
    messages = [
        SystemMessage(content=SEARCH_SYSTEM),
        HumanMessage(content=f"Research query: {query}"),
    ]

    all_results: list[SearchResult] = []
    queries_used: list[str] = []
    tool_calls_log: list[dict[str, Any]] = []

    # Ask LLM for search queries
    response = _llm.invoke(messages)
    raw = response.content.strip()

    # Parse the search queries
    search_queries = [query]  # fallback to original query
    try:
        if "```" in raw:
            raw = raw.split("```")[1].strip().lstrip("json").strip()
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "queries" in parsed:
            search_queries = parsed["queries"][:3]
    except (json.JSONDecodeError, KeyError):
        pass  # use fallback

    # Execute each search query
    for sq in search_queries:
        queries_used.append(sq)
        try:
            results = duckduckgo_search.invoke({"query": sq})
            tool_calls_log.append({
                "agent": "search_agent",
                "tool": "duckduckgo_search",
                "args": {"query": sq},
                "result_count": len(results) if isinstance(results, list) else 1,
            })

            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict) and r.get("url"):
                        all_results.append(SearchResult(
                            title=r.get("title", ""),
                            url=r.get("url", ""),
                            snippet=r.get("snippet", ""),
                        ))
        except Exception:
            pass  # skip failed queries, continue with others

    # Deduplicate results by URL
    seen_urls: set[str] = set()
    unique_results: list[SearchResult] = []
    for r in all_results:
        if r["url"] and r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique_results.append(r)

    return {
        "search_results":      unique_results,
        "search_queries_used": queries_used,
        "tool_calls_log":      tool_calls_log,
        "messages":            [AIMessage(content=f"[search_agent] Found {len(unique_results)} results for: {query}")],
    }

