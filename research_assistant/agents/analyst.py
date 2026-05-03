"""
agents/analyst.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analyst Agent — deep reasoning over summaries.
Has access to the Python REPL for calculations / data work.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import json
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

from state import ResearchState
from tools.python_repl import REPL_TOOLS
from config.settings import GROQ_API_KEY, GROQ_MODEL

_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.3)
_llm_with_repl = _llm.bind_tools(REPL_TOOLS)
_TOOL_MAP = {t.name: t for t in REPL_TOOLS}

ANALYST_SYSTEM = """You are a senior research analyst. Your job is to:
  1. Synthesise the provided summaries into a coherent, deep analysis.
  2. Identify gaps, contradictions, or nuances in the evidence.
  3. Use the python_repl tool if any calculations, statistics, or data processing would strengthen the analysis.
  4. Structure your final analysis with clear sections:
       - Key Findings
       - Supporting Evidence
       - Gaps / Caveats
       - Conclusion

Be analytical, not just descriptive. Explain WHY things are the way they are."""


def analyst_node(state: ResearchState) -> dict:
    """Analyst node — synthesises summaries into deep analysis."""
    query     = state["query"]
    summaries = state.get("summaries", [])

    summaries_text = "\n\n---\n\n".join(summaries) if summaries else "No summaries available."

    messages: list = [
        SystemMessage(content=ANALYST_SYSTEM),
        HumanMessage(content=f"Research Query: {query}\n\nSummaries:\n{summaries_text}\n\nProvide a deep analysis."),
    ]

    tool_calls_log: list[dict[str, Any]] = []
    key_facts: list[str] = []

    # Allow one round of REPL tool calls
    response = _llm_with_repl.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tc in response.tool_calls:
            tool_fn = _TOOL_MAP.get(tc["name"])
            if tool_fn:
                result = tool_fn.invoke(tc["args"])
                tool_calls_log.append({
                    "agent": "analyst",
                    "tool":  tc["name"],
                    "args":  tc["args"],
                    "result": str(result)[:500],
                })
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                ))

        # Get final analysis after tool results
        final_response = _llm.invoke(messages)
        analysis = final_response.content.strip()
    else:
        analysis = response.content.strip()

    # Extract key facts (lines starting with - or •)
    for line in analysis.split("\n"):
        line = line.strip()
        if line.startswith(("-", "•", "*")) and len(line) > 10:
            key_facts.append(line.lstrip("-•* ").strip())

    return {
        "analysis":      analysis,
        "key_facts":     key_facts[:10],  # top 10
        "tool_calls_log": tool_calls_log,
        "messages":      [AIMessage(content=f"[analyst] Analysis complete. {len(key_facts)} key facts extracted.")],
    }
