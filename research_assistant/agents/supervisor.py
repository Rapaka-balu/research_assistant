"""
agents/supervisor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The Supervisor is the brain of the graph. It:
  1. Reads the current ResearchState
  2. Decides which agent should act next
  3. Writes `next_agent` into state (the conditional edge reads this)

Routing logic:
  query arrives        → search_agent
  search done          → summarizer
  summaries ready      → analyst
  analysis done        → citation_agent
  final_answer set     → __end__
  iteration >= MAX     → citation_agent (force finish)

The LLM is used to make the routing decision so it can handle
edge cases (e.g. search returned nothing → go straight to analyst).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState
from config.settings import (
    GROQ_API_KEY, GROQ_MODEL, MAX_ITERATIONS,
    AGENT_SEARCH, AGENT_SUMMARIZER, AGENT_ANALYST,
    AGENT_CITATION, AGENT_END, VALID_AGENTS,
)

_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)

SUPERVISOR_SYSTEM = """You are a research supervisor managing a team of specialist agents.

Your ONLY job is to decide which agent should act next given the current research state.
Reply with a single JSON object: {{"next": "<agent_name>"}}

Available agents:
  - search_agent    : searches the web for information
  - summarizer      : summarises raw search results into concise points
  - analyst         : synthesises summaries into deep analysis with reasoning
  - citation_agent  : writes the final answer with proper citations
  - __end__         : stop — the final answer is complete

Routing rules:
  1. If search_results is empty → search_agent
  2. If search_results exist but summaries is empty → summarizer
  3. If summaries exist but analysis is None → analyst
  4. If analysis exists but final_answer is None → citation_agent
  5. If final_answer is set → __end__
  6. If iteration >= {max_iter} → citation_agent (force completion)

Reply ONLY with the JSON. No explanation."""


def supervisor_node(state: ResearchState) -> dict:
    """
    Supervisor node — reads state, decides next agent.
    Returns a partial state update: only `next_agent` and `iteration`.
    """
    iteration = state.get("iteration", 0)

    # Hard guard — force finish if too many loops
    if iteration >= MAX_ITERATIONS:
        return {"next_agent": AGENT_CITATION, "iteration": iteration + 1}

    # If final answer already written, end
    if state.get("final_answer"):
        return {"next_agent": AGENT_END, "iteration": iteration + 1}

    prompt = SUPERVISOR_SYSTEM.format(max_iter=MAX_ITERATIONS)

    # Build a concise state summary for the LLM
    state_summary = {
        "query":              state.get("query", ""),
        "iteration":          iteration,
        "search_results_n":   len(state.get("search_results", [])),
        "summaries_n":        len(state.get("summaries", [])),
        "analysis_present":   state.get("analysis") is not None,
        "final_answer_present": state.get("final_answer") is not None,
    }

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Current state:\n{json.dumps(state_summary, indent=2)}\n\nWhat should happen next?"),
    ]

    response = _llm.invoke(messages)
    raw = response.content.strip()

    # Parse JSON response
    try:
        # Handle markdown code fences if model wraps JSON
        if "```" in raw:
            raw = raw.split("```")[1].strip().lstrip("json").strip()
        decision = json.loads(raw)
        next_agent = decision.get("next", AGENT_SEARCH)
    except (json.JSONDecodeError, KeyError):
        # Fallback: derive from state deterministically
        next_agent = _fallback_route(state)

    # Validate — never route to an unknown node
    if next_agent not in VALID_AGENTS:
        next_agent = _fallback_route(state)

    return {"next_agent": next_agent, "iteration": iteration + 1}


def _fallback_route(state: ResearchState) -> str:
    """Deterministic fallback routing when LLM output is unparseable."""
    if not state.get("search_results"):
        return AGENT_SEARCH
    if not state.get("summaries"):
        return AGENT_SUMMARIZER
    if not state.get("analysis"):
        return AGENT_ANALYST
    if not state.get("final_answer"):
        return AGENT_CITATION
    return AGENT_END


def route_after_supervisor(state: ResearchState) -> str:
    """
    Conditional edge function — called by LangGraph after the supervisor node.
    Returns the name of the next node to execute.
    """
    return state.get("next_agent", AGENT_SEARCH)
