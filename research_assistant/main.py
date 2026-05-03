"""
main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Entry point for the Multi-Agent Research Assistant.
Features:
  - Interactive CLI with session persistence (thread_id)
  - Displays agent activity in real time as the graph executes
  - Streaming-style step output using LangGraph stream()
  - Session listing and resumption
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import sys
import os

# Fix Windows console encoding for Unicode/emoji
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import uuid
import argparse
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from graph import build_graph, get_initial_state
from memory.checkpointer import list_sessions, get_checkpointer
from config.settings import GROQ_API_KEY

console = Console()

AGENT_COLORS = {
    "supervisor":    "yellow",
    "search_agent":  "cyan",
    "summarizer":    "blue",
    "analyst":       "magenta",
    "citation_agent": "green",
}

AGENT_ICONS = {
    "supervisor":    "[>]",
    "search_agent":  "[S]",
    "summarizer":    "[M]",
    "analyst":       "[A]",
    "citation_agent": "[C]",
}


def check_config():
    """Validate required config before running."""
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        console.print(Panel(
            "[red]GROQ_API_KEY not set.[/red]\n\n"
            "1. Copy [bold].env.example[/bold] to [bold].env[/bold]\n"
            "2. Get a free key at [link=https://console.groq.com]console.groq.com[/link]\n"
            "3. Add it to .env",
            title="⚙ Setup Required", border_style="red"
        ))
        sys.exit(1)


def run_research(query: str, thread_id: str, graph) -> str:
    """
    Run the research graph on a query.
    Streams node-by-node updates to the console.
    Returns the final answer.
    """
    initial_state = get_initial_state(query)
    config = {"configurable": {"thread_id": thread_id}}

    console.print(f"\n[dim]Session: {thread_id}[/dim]")
    console.rule("[bold]Agent Activity[/bold]")

    final_answer = "No answer produced."

    # Stream mode: yields {"node_name": state_update} after each node
    for step in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, update in step.items():
            if node_name == "__end__":
                continue

            color = AGENT_COLORS.get(node_name, "white")
            icon  = AGENT_ICONS.get(node_name, "•")

            # Show what each agent did
            status_parts = []
            if update.get("search_results"):
                status_parts.append(f"{len(update['search_results'])} results found")
            if update.get("summaries"):
                status_parts.append("summaries ready")
            if update.get("analysis"):
                status_parts.append("analysis complete")
            if update.get("final_answer"):
                status_parts.append("final answer ready")
                final_answer = update["final_answer"]
            if update.get("next_agent"):
                status_parts.append(f"→ routing to {update['next_agent']}")

            status = "  |  ".join(status_parts) if status_parts else "processing..."
            console.print(f"  [{color}]{icon} {node_name}[/{color}]  [dim]{status}[/dim]")

    return final_answer


def interactive_session(graph, thread_id: str):
    """Run an interactive multi-turn research session."""
    console.print(Panel(
        f"[bold green]Multi-Agent Research Assistant[/bold green]\n"
        f"[dim]Groq · LangGraph · DuckDuckGo · RAGAS[/dim]\n\n"
        f"Type your research question. Session persists between runs.\n"
        f"Commands: [bold]exit[/bold] · [bold]new[/bold] (new session) · [bold]sessions[/bold]",
        border_style="green"
    ))

    while True:
        try:
            query = console.input("\n[bold cyan]Research Query >[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not query:
            continue
        if query.lower() == "exit":
            console.print("[dim]Goodbye.[/dim]")
            break
        if query.lower() == "new":
            thread_id = f"session-{uuid.uuid4().hex[:8]}"
            console.print(f"[green]New session started: {thread_id}[/green]")
            continue
        if query.lower() == "sessions":
            cp = get_checkpointer()
            sessions = list_sessions(cp)
            if sessions:
                t = Table(title="Saved Sessions")
                t.add_column("Thread ID", style="cyan")
                for s in sessions:
                    t.add_row(s)
                console.print(t)
            else:
                console.print("[dim]No saved sessions yet.[/dim]")
            continue

        # Run the research
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            progress.add_task("Running agents...", total=None)
            answer = run_research(query, thread_id, graph)

        console.rule("[bold]Answer[/bold]")
        console.print(Markdown(answer))
        console.rule()


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Research Assistant")
    parser.add_argument("--query",   "-q", type=str, help="One-shot query (non-interactive)")
    parser.add_argument("--session", "-s", type=str, help="Resume a session by thread_id")
    parser.add_argument("--eval",    "-e", action="store_true", help="Run RAGAS evaluation")
    parser.add_argument("--eval-n",        type=int, default=5, help="Number of eval questions")
    args = parser.parse_args()

    check_config()

    if args.eval:
        from evals.evaluate import run_evaluation
        run_evaluation(max_questions=args.eval_n)
        return

    # Build graph with memory
    console.print("[dim]Building graph...[/dim]", end="\r")
    graph = build_graph(with_memory=True)
    console.print("[dim]Graph ready.     [/dim]")

    thread_id = args.session or f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if args.query:
        # One-shot mode
        answer = run_research(args.query, thread_id, graph)
        console.rule("[bold]Answer[/bold]")
        console.print(Markdown(answer))
    else:
        # Interactive mode
        interactive_session(graph, thread_id)


if __name__ == "__main__":
    main()
