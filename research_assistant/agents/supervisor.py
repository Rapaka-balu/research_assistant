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

import re
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from state import ResearchState
from config.settings import (
    GROQ_API_KEY, GROQ_MODEL, MAX_ITERATIONS,
    AGENT_SEARCH, AGENT_SUMMARIZER, AGENT_ANALYST,
    AGENT_CITATION, AGENT_END, VALID_AGENTS,
)

_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)

# ── Greeting / casual message detection ───────────────────────────────
# Patterns that indicate a greeting or casual message, NOT a research query.
_GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey|howdy|hola|namaste|yo)\s*[!.,?\s]*$",
    r"^\s*(good\s*(morning|afternoon|evening|day|night))\s*[!.,?\s]*$",
    r"^\s*(what'?s\s*up|sup|hiya|greetings|salutations)\s*[!.,?\s]*$",
    r"^\s*(thanks?(\s*you)?|thank\s*u|thx|ty)\s*[!.,?\s]*$",
    r"^\s*(bye|goodbye|see\s*ya|later|cya)\s*[!.,?\s]*$",
    r"^\s*(how\s*are\s*you|how\s*r\s*u|how\s*do\s*you\s*do)\s*[!.,?\s]*$",
    r"^\s*(who\s*are\s*you|what\s*are\s*you|what\s*can\s*you\s*do)\s*[!.,?\s]*$",
    r"^\s*(help|help\s*me)\s*[!.,?\s]*$",
]
_GREETING_RE = re.compile("|".join(_GREETING_PATTERNS), re.IGNORECASE)

# Friendly responses keyed by intent category
_GREETING_RESPONSES = {
    "greeting": (
        "👋 Hello! I'm your **Multi-Agent Research Assistant**.\n\n"
        "I can help you research any topic in depth. Just ask me a question and "
        "my team of AI agents will:\n"
        "- 🔍 **Search** the web for relevant information\n"
        "- 📝 **Summarize** the key findings\n"
        "- 🧠 **Analyze** the evidence with deep reasoning\n"
        "- 📚 **Cite** sources in a polished final answer\n\n"
        "**Try asking something like:**\n"
        '- *"What are the latest breakthroughs in quantum computing?"*\n'
        '- *"Compare renewable energy sources for home use"*\n'
        '- *"Explain the economic impact of AI on the job market"*\n\n'
        "What would you like to research today?"
    ),
    "thanks": (
        "You're welcome! 😊 If you have another research question, feel free to ask. "
        "I'm always ready to help you explore new topics."
    ),
    "farewell": (
        "Goodbye! 👋 It was great helping you. Come back anytime you need research assistance!"
    ),
    "identity": (
        "I'm your **Multi-Agent Research Assistant** — a system powered by multiple "
        "specialized AI agents working together.\n\n"
        "Here's how I work:\n"
        "1. **Supervisor** — coordinates the research workflow\n"
        "2. **Search Agent** — finds relevant information from the web\n"
        "3. **Summarizer** — distills search results into key facts\n"
        "4. **Analyst** — synthesizes findings with deep reasoning\n"
        "5. **Citation Agent** — produces a polished answer with references\n\n"
        "Ask me any research question to get started!"
    ),
    "help": (
        "🆘 **How to use the Research Assistant:**\n\n"
        "Simply type a research question and I'll handle the rest! My agents will "
        "search the web, summarize findings, analyze the evidence, and deliver a "
        "well-cited answer.\n\n"
        "**Tips for best results:**\n"
        "- Be specific in your question\n"
        "- Ask about factual, researchable topics\n"
        "- Use follow-up questions to dig deeper\n\n"
        "**Commands:** `exit` · `new` (new session) · `sessions`"
    ),
    "how_are_you": (
        "I'm doing great, thank you for asking! 😊 I'm ready and fully operational. "
        "Feel free to ask me any research question — I'm here to help you find "
        "well-researched, cited answers on any topic."
    ),
}


def _detect_casual_message(query: str) -> str | None:
    """
    Detect if a query is a greeting or casual message instead of a research question.
    Returns a response category key if matched, or None for research queries.
    """
    if not _GREETING_RE.match(query):
        return None

    q = query.strip().lower().rstrip("!.,? ")

    if re.match(r"(thanks?(\s*you)?|thank\s*u|thx|ty)", q):
        return "thanks"
    if re.match(r"(bye|goodbye|see\s*ya|later|cya)", q):
        return "farewell"
    if re.match(r"(who\s*are\s*you|what\s*are\s*you|what\s*can\s*you\s*do)", q):
        return "identity"
    if re.match(r"(help|help\s*me)", q):
        return "help"
    if re.match(r"(how\s*are\s*you|how\s*r\s*u|how\s*do\s*you\s*do)", q):
        return "how_are_you"
    return "greeting"


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

    On the first iteration, detects greetings and casual messages to
    respond directly without invoking the research pipeline.
    """
    iteration = state.get("iteration", 0)

    # Hard guard — force finish if too many loops
    if iteration >= MAX_ITERATIONS:
        return {"next_agent": AGENT_CITATION, "iteration": iteration + 1}

    # If final answer already written, end
    if state.get("final_answer"):
        return {"next_agent": AGENT_END, "iteration": iteration + 1}

    # ── Greeting / casual message detection (first pass only) ─────────
    if iteration == 0:
        query = state.get("query", "")
        category = _detect_casual_message(query)
        if category is not None:
            friendly_reply = _GREETING_RESPONSES.get(category, _GREETING_RESPONSES["greeting"])
            return {
                "next_agent":   AGENT_END,
                "iteration":    iteration + 1,
                "final_answer": friendly_reply,
                "messages":     [AIMessage(content=f"[supervisor] Greeting detected — responded directly.")],
            }

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
