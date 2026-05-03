# Multi-Agent Research Assistant — LangGraph + Groq

## Architecture

```
User Query
    │
    ▼
[Supervisor Agent]  ←── decides which agent to call next
    │
    ├──▶ [Search Agent]      — DuckDuckGo web search + scraping
    ├──▶ [Summarizer Agent]  — condenses search results
    ├──▶ [Analyst Agent]     — synthesizes across sources, adds reasoning
    └──▶ [Citation Agent]    — formats final answer with citations
            │
            ▼
    [State Graph / Memory]  ←── LangGraph checkpointing (SQLite)
            │
            ▼
    [RAGAS Evaluator]       ←── faithfulness, relevancy, context precision
```

## Key Concepts Demonstrated
- **State management**: TypedDict state schema with typed reducers
- **Checkpointed memory**: SQLite-backed persistence across sessions
- **Agent communication**: message passing via Annotated reducers
- **Tool calling**: DuckDuckGo search + Python REPL tools
- **Prompt engineering**: system prompts per agent with role/task separation
- **LLM Evals**: RAGAS metrics over agent outputs

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in GROQ_API_KEY in .env

python main.py
# Or run evals:
python evals/evaluate.py
```

## Project Structure
```
research_assistant/
├── main.py                  # Entry point — CLI interface
├── graph.py                 # LangGraph graph definition & compilation
├── state.py                 # Shared TypedDict state schema
├── config/
│   └── settings.py          # Env vars, model config
├── agents/
│   ├── supervisor.py        # Supervisor — routes to next agent
│   ├── search_agent.py      # Web search + tool calling
│   ├── summarizer.py        # Summarizes retrieved content
│   ├── analyst.py           # Synthesizes + reasons across sources
│   └── citation_agent.py   # Formats final cited response
├── tools/
│   ├── search_tool.py       # DuckDuckGo search wrapper
│   └── python_repl.py       # Python REPL tool (safe exec)
├── memory/
│   └── checkpointer.py      # SQLite checkpointer + session management
├── evals/
│   ├── evaluate.py          # RAGAS evaluation runner
│   └── benchmark_dataset.py # 20-question benchmark Q&A pairs
└── requirements.txt
```
