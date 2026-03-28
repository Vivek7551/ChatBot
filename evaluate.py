"""
evaluate.py
-----------
Runs the chatbot against a set of benchmark questions drawn directly from
the project spec and outputs a report showing how well it answers.

Useful for:
  • Demonstrating the chatbot to Prof. Devreotes on Monday
  • Comparing RAG vs GraphRAG performance later
  • Identifying gaps in the corpus

Usage:
    python evaluate.py
"""

import json
import os
import time
from rag_engine import RAGEngine


# ---------------------------------------------------------------------------
# Benchmark questions — drawn from the project spec + paper content
# ---------------------------------------------------------------------------
QUESTIONS = [
    # Project spec questions
    {
        "category": "Research themes",
        "question": "What are Prof. Devreotes' main research themes in this corpus?",
    },
    {
        "category": "Foundational papers",
        "question": "Which papers are foundational for understanding acetylcholine receptor metabolism?",
    },
    {
        "category": "Methods",
        "question": "What experimental methods recur across these papers?",
    },
    {
        "category": "Collaborators",
        "question": "Which collaborators appear most often in this body of work?",
    },
    {
        "category": "Starting point",
        "question": "Which paper should a newcomer read first to understand this research area?",
    },
    # Specific scientific questions
    {
        "category": "Specific finding",
        "question": "What is the half-life of acetylcholine receptors in chick myotubes?",
    },
    {
        "category": "Subcellular location",
        "question": "Where are newly synthesized acetylcholine receptors located before they reach the plasma membrane?",
    },
    {
        "category": "Mechanism",
        "question": "How does receptor degradation occur — what cellular process is involved?",
    },
    {
        "category": "De novo synthesis",
        "question": "What evidence shows that new receptors are synthesized de novo rather than recycled?",
    },
    {
        "category": "Evolution over time",
        "question": "How did Devreotes' understanding of the receptor precursor pool change across these papers?",
    },
    {
        "category": "Cross-paper connection",
        "question": "How does the 1975 turnover paper connect to the 1978 Golgi apparatus paper?",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_evaluation(output_file: str = "data/evaluation_results.json") -> None:
    print("=" * 70)
    print("  Devreotes RAG Chatbot — Evaluation")
    print("=" * 70)

    engine = RAGEngine()
    engine.load()

    results = []

    for i, item in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/{len(QUESTIONS)}] Category: {item['category']}")
        print(f"  Q: {item['question']}")
        print("  Answering...", end="", flush=True)

        start = time.time()
        # Each question is independent (no conversation history)
        engine.reset_conversation()

        try:
            answer, chunks = engine.ask(item["question"], multi_turn=False)
            elapsed = time.time() - start

            # Source attribution
            sources = list({
                c["source"]: f"{c['title']} ({c['year']})"
                for c in chunks
            }.values())

            print(f"\r  A: {answer[:120]}{'...' if len(answer) > 120 else ''}")
            print(f"     Sources: {' | '.join(s[:40] for s in sources)}")
            print(f"     Time: {elapsed:.1f}s")

            results.append({
                "category": item["category"],
                "question": item["question"],
                "answer": answer,
                "sources_cited": sources,
                "top_chunk_scores": [c["score"] for c in chunks],
                "elapsed_seconds": round(elapsed, 2),
                "status": "ok",
            })

        except Exception as e:
            print(f"\r  ERROR: {e}")
            results.append({
                "category": item["category"],
                "question": item["question"],
                "answer": None,
                "error": str(e),
                "status": "error",
            })

    # Save results
    os.makedirs("data", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{'=' * 70}")
    print(f"  Completed: {ok}/{len(QUESTIONS)} questions answered successfully")
    print(f"  Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
