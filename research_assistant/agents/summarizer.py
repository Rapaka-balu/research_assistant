"""
agents/summarizer.py
Condenses raw search results into clean bullet-point summaries.
"""
from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from state import ResearchState
from config.settings import GROQ_API_KEY, GROQ_MODEL

_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.2)

SUMMARIZER_SYSTEM = """You are a precise research summarizer.

Given raw search results, extract the most relevant facts for the query.
Format your output as a numbered list of concise factual statements.
Each point should be:
  - Self-contained (makes sense without context)
  - Factual (no opinions, no speculation)
  - Sourced (mention the source URL briefly when relevant)

Maximum 8 bullet points. Be concise but information-dense."""


def summarizer_node(state: ResearchState) -> dict:
    """Summarize raw search results into structured bullet points."""
    query   = state["query"]
    results = state.get("search_results", [])

    if not results:
        return {
            "summaries": ["No search results were found. Proceeding with general knowledge."],
            "messages":  [AIMessage(content="[summarizer] No results to summarize.")],
        }

    # Format results for the LLM
    results_text = "\n\n".join([
        f"[{i+1}] {r['title']}\nURL: {r['url']}\n{r['snippet']}"
        for i, r in enumerate(results[:10])  # cap at 10
    ])

    messages = [
        SystemMessage(content=SUMMARIZER_SYSTEM),
        HumanMessage(content=f"Query: {query}\n\nSearch Results:\n{results_text}\n\nSummarise the key facts."),
    ]

    response = _llm.invoke(messages)
    summary = response.content.strip()

    return {
        "summaries": [summary],
        "messages":  [AIMessage(content=f"[summarizer] Summarised {len(results)} results.")],
    }
