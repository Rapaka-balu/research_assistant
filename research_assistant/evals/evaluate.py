"""
evals/evaluate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAGAS-based evaluation of the research agent.

Metrics evaluated:
  - faithfulness        : does the answer only contain claims supported
                          by the retrieved context?
  - answer_relevancy    : does the answer actually address the question?
  - context_precision   : is the retrieved context relevant (not noisy)?
  - context_recall      : does the context cover the ground truth?

How it works:
  1. Run the agent on each benchmark question
  2. Collect: question, answer, contexts (search snippets), ground truth
  3. Pass to RAGAS evaluate() which returns scores per metric
  4. Print a summary report

Note on RAGAS judge model:
  RAGAS uses an LLM to compute faithfulness and answer_relevancy.
  By default it uses GPT-4. Here we override it to use Groq (free).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq

from graph import build_graph, get_initial_state
from config.settings import GROQ_API_KEY, GROQ_MODEL
from evals.benchmark_dataset import BENCHMARK_QA


def run_agent_on_question(graph, query: str, thread_id: str) -> dict:
    """Run the graph on one question and return eval-ready dict."""
    initial_state = get_initial_state(query)
    config = {"configurable": {"thread_id": thread_id}}

    final_state = graph.invoke(initial_state, config=config)

    # Collect contexts = all search snippets retrieved
    contexts = [
        r["snippet"] for r in final_state.get("search_results", [])
        if r.get("snippet")
    ]

    return {
        "question": query,
        "answer":   final_state.get("final_answer", "No answer produced."),
        "contexts": contexts[:5] if contexts else ["No context retrieved."],
    }


def build_ragas_dataset(graph, benchmark: list[dict], max_questions: int = 10) -> Dataset:
    """
    Run agent on benchmark questions and build a HuggingFace Dataset
    in the format RAGAS expects.
    """
    rows = {
        "question":    [],
        "answer":      [],
        "contexts":    [],
        "ground_truth": [],
    }

    print(f"\n Running agent on {min(max_questions, len(benchmark))} benchmark questions...")
    for i, item in enumerate(benchmark[:max_questions]):
        print(f"  [{i+1}/{max_questions}] {item['question'][:60]}...")
        try:
            result = run_agent_on_question(graph, item["question"], f"eval-{i}")
            rows["question"].append(result["question"])
            rows["answer"].append(result["answer"])
            rows["contexts"].append(result["contexts"])
            rows["ground_truth"].append(item["ground_truth"])
        except Exception as e:
            print(f"    ⚠ Error on question {i+1}: {e}")
            # Add placeholder row so dataset stays aligned
            rows["question"].append(item["question"])
            rows["answer"].append(f"Error: {e}")
            rows["contexts"].append(["Error retrieving context."])
            rows["ground_truth"].append(item["ground_truth"])

    return Dataset.from_dict(rows)


def run_evaluation(max_questions: int = 5):
    """
    Full evaluation pipeline:
      1. Build graph (no memory for evals — fresh state per question)
      2. Run agent on benchmark
      3. Evaluate with RAGAS
      4. Print report
    """
    print("━" * 60)
    print(" RAGAS Evaluation — Multi-Agent Research Assistant")
    print("━" * 60)

    # Build graph without persistent memory for evals
    graph = build_graph(with_memory=False)

    # Build eval dataset
    dataset = build_ragas_dataset(graph, BENCHMARK_QA, max_questions=max_questions)

    # Configure RAGAS to use Groq instead of OpenAI
    groq_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
    judge_llm = LangchainLLMWrapper(groq_llm)

    print("\n Computing RAGAS metrics (faithfulness, relevancy, precision, recall)...")
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = judge_llm

    results = evaluate(dataset=dataset, metrics=metrics)

    # ── Print report ──────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print(" EVALUATION RESULTS")
    print("━" * 60)
    scores = results.to_pandas()

    print(f"\n  Faithfulness        (0–1):  {scores['faithfulness'].mean():.3f}")
    print(f"  Answer Relevancy    (0–1):  {scores['answer_relevancy'].mean():.3f}")
    print(f"  Context Precision   (0–1):  {scores['context_precision'].mean():.3f}")
    print(f"  Context Recall      (0–1):  {scores['context_recall'].mean():.3f}")
    print("\n  Score = 1.0 is perfect. Aim for > 0.7 on all metrics.")
    print("━" * 60)

    # Save to CSV for analysis
    scores.to_csv("evals/eval_results.csv", index=False)
    print(f"\n  Full results saved to evals/eval_results.csv")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of benchmark questions to eval (default: 5)")
    args = parser.parse_args()
    run_evaluation(max_questions=args.n)
