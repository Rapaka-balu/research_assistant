"""
graph.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LangGraph graph definition.

Graph structure:
                    ┌─────────────────────────────────────┐
                    │                                     │
  START ──► supervisor ──► search_agent  ──┐             │
               ▲           summarizer   ──┤             │
               │           analyst      ──┤─► supervisor (loop)
               │           citation_agent ─┘             │
               │                                         │
               └───────────────── __end__ ◄──────────────┘

The supervisor decides routing via the `next_agent` state field.
`route_after_supervisor` is the conditional edge that reads it.

Compiling with checkpointer enables automatic SQLite persistence.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from state import ResearchState
from agents.supervisor import supervisor_node, route_after_supervisor
from agents.search_agent import search_agent_node
from agents.summarizer import summarizer_node
from agents.analyst import analyst_node
from agents.citation_agent import citation_agent_node
from memory.checkpointer import get_checkpointer
from config.settings import (
    AGENT_SEARCH, AGENT_SUMMARIZER, AGENT_ANALYST, AGENT_CITATION, AGENT_END
)


def build_graph(with_memory: bool = True):
    """
    Builds and compiles the research graph.

    Args:
        with_memory: If True, attach SQLite checkpointer for persistence.

    Returns:
        Compiled LangGraph CompiledGraph instance.
    """
    # ── Define the graph with our state schema ────────────────────────
    builder = StateGraph(ResearchState)

    # ── Register all nodes ────────────────────────────────────────────
    builder.add_node("supervisor",     supervisor_node)
    builder.add_node(AGENT_SEARCH,     search_agent_node)
    builder.add_node(AGENT_SUMMARIZER, summarizer_node)
    builder.add_node(AGENT_ANALYST,    analyst_node)
    builder.add_node(AGENT_CITATION,   citation_agent_node)

    # ── Entry point: always start at supervisor ───────────────────────
    builder.add_edge(START, "supervisor")

    # ── Conditional routing from supervisor ───────────────────────────
    # route_after_supervisor reads state["next_agent"] and returns
    # the name of the next node — LangGraph follows that edge.
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            AGENT_SEARCH:     AGENT_SEARCH,
            AGENT_SUMMARIZER: AGENT_SUMMARIZER,
            AGENT_ANALYST:    AGENT_ANALYST,
            AGENT_CITATION:   AGENT_CITATION,
            AGENT_END:        END,
        }
    )

    # ── All worker agents loop back to supervisor ─────────────────────
    builder.add_edge(AGENT_SEARCH,     "supervisor")
    builder.add_edge(AGENT_SUMMARIZER, "supervisor")
    builder.add_edge(AGENT_ANALYST,    "supervisor")
    builder.add_edge(AGENT_CITATION,   "supervisor")

    # ── Compile ───────────────────────────────────────────────────────
    if with_memory:
        checkpointer = get_checkpointer()
        return builder.compile(checkpointer=checkpointer)
    else:
        return builder.compile()


def get_initial_state(query: str) -> dict:
    """
    Returns a clean initial state dict for a new query.
    All list fields start empty; reducers will accumulate into them.
    """
    return {
        "query":               query,
        "messages":            [],
        "next_agent":          "",
        "iteration":           0,
        "search_results":      [],
        "search_queries_used": [],
        "summaries":           [],
        "analysis":            None,
        "key_facts":           [],
        "final_answer":        None,
        "citations":           [],
        "tool_calls_log":      [],
    }
