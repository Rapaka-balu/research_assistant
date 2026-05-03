"""
agents/citation_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Citation Agent — produces the final user-facing answer.
Weaves the analysis together with structured citations.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from state import ResearchState
from config.settings import GROQ_API_KEY, GROQ_MODEL

_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.4)

CITATION_SYSTEM = """You are a professional research writer. Produce the final answer to the user's query.

Your response MUST:
  1. Directly answer the question in the opening paragraph.
  2. Use inline citation markers like [1], [2] when referencing sources.
  3. Include a "References" section at the end listing numbered sources.
  4. Be clear, well-structured, and appropriately detailed (not too long).
  5. Use markdown formatting (headers, bullets where helpful).

Format:
## Answer
<direct answer>

## Details
<supporting information with [N] citations>

## References
[1] Title — URL
[2] Title — URL
..."""


def citation_agent_node(state: ResearchState) -> dict:
    """
    Citation agent — assembles final answer from analysis + search results.
    """
    query    = state["query"]
    analysis = state.get("analysis", "")
    results  = state.get("search_results", [])
    summaries = state.get("summaries", [])

    # Build sources list for the LLM
    sources_text = "\n".join([
        f"[{i+1}] {r['title']} — {r['url']}\n    {r['snippet'][:200]}"
        for i, r in enumerate(results[:8])
    ])

    context = f"""Query: {query}

Analysis:
{analysis}

Summaries:
{chr(10).join(summaries)}

Available Sources:
{sources_text}"""

    messages = [
        SystemMessage(content=CITATION_SYSTEM),
        HumanMessage(content=context),
    ]

    response = _llm.invoke(messages)
    final_answer = response.content.strip()

    # Extract citations from the References section
    citations: list[dict[str, str]] = []
    ref_section = re.search(r"## References\n(.*?)$", final_answer, re.DOTALL)
    if ref_section:
        for line in ref_section.group(1).strip().split("\n"):
            match = re.match(r"\[(\d+)\]\s+(.+?)\s+[—–-]+\s+(https?://\S+)", line)
            if match:
                citations.append({
                    "index": match.group(1),
                    "title": match.group(2).strip(),
                    "url":   match.group(3).strip(),
                })

    return {
        "final_answer": final_answer,
        "citations":    citations,
        "messages":     [AIMessage(content=f"[citation_agent] Final answer ready. {len(citations)} citations.")],
    }
