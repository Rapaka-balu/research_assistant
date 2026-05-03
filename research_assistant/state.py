"""
state.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Defines the SHARED STATE that flows through every node in the graph.

Key LangGraph concepts demonstrated here:
  - TypedDict schema  → every field is typed and validated
  - Annotated reducer → operator.add means messages ACCUMULATE
                        (append-only); a plain type would OVERWRITE
  - Optional fields   → agents only populate what they produce

The supervisor reads `next_agent` to decide routing.
Each agent reads what it needs and writes its output field.
All agents append to `messages` — this is the conversation memory.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class SearchResult(TypedDict):
    """One search result from DuckDuckGo."""
    title: str
    url: str
    snippet: str


class ResearchState(TypedDict):
    """
    Shared state schema for the multi-agent research graph.

    Reducer rules:
      messages       → operator.add  (append every new message)
      search_results → operator.add  (accumulate results across calls)
      all others     → default (last-write-wins)
    """

    # ── Conversation memory (append-only via reducer) ─────────────────
    messages: Annotated[list[BaseMessage], operator.add]

    # ── User input ────────────────────────────────────────────────────
    query: str                          # original user question

    # ── Supervisor routing ────────────────────────────────────────────
    next_agent: str                     # which node to call next
    iteration: int                      # guards against infinite loops

    # ── Search agent outputs ──────────────────────────────────────────
    search_results: Annotated[list[SearchResult], operator.add]
    search_queries_used: Annotated[list[str], operator.add]

    # ── Summarizer outputs ────────────────────────────────────────────
    summaries: Annotated[list[str], operator.add]

    # ── Analyst outputs ───────────────────────────────────────────────
    analysis: Optional[str]             # synthesised reasoning
    key_facts: Annotated[list[str], operator.add]

    # ── Citation agent outputs ────────────────────────────────────────
    final_answer: Optional[str]         # formatted answer with citations
    citations: Annotated[list[dict[str, str]], operator.add]

    # ── Tool call trace (for evals / debugging) ───────────────────────
    tool_calls_log: Annotated[list[dict[str, Any]], operator.add]
