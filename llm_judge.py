"""
llm_judge.py
------------
Iterates through a Golden Set of questions and evaluates the chatbot's answers using
an LLM-as-a-Judge pattern against the ground truth.
"""

import json
import os
import time
from rag_engine import RAGEngine

GOLDEN_SET = [
    {
        "question": "What is the half-life of acetylcholine receptors in chick myotubes?",
        "ground_truth": "The half-life of acetylcholine receptors in chick myotubes is approximately 17 hours."
    },
    {
        "question": "Where are newly synthesized acetylcholine receptors located before they reach the plasma membrane?",
        "ground_truth": "They are located in the Golgi apparatus."
    },
    {
        "question": "How does receptor degradation occur — what cellular process is involved?",
        "ground_truth": "Receptor degradation occurs via internalization followed by lysosomal degradation."
    }
]

def run_judge_evaluation(output_file: str = "data/judge_results.json") -> None:
    print("=" * 70)
    print("  Devreotes RAG Chatbot — LLM-as-a-Judge Evaluation")
    print("=" * 70)

    engine = RAGEngine()
    engine.load()

    results = []

    for i, item in enumerate(GOLDEN_SET, 1):
        print(f"\n[{i}/{len(GOLDEN_SET)}] Question: {item['question']}")
        print("  Ground Truth:", item['ground_truth'])
        print("  Generating answer and evaluating...", end="", flush=True)

        start = time.time()
        engine.reset_conversation()

        try:
            # 1. Process: Run through current RAG pipeline
            answer, chunks = engine.ask(item["question"], multi_turn=False)
            
            # 2. Evaluate: Send to scoring LLM
            judge_evaluation = engine.evaluate_with_judge(
                question=item["question"],
                chatbot_answer=answer,
                ground_truth=item["ground_truth"]
            )
            
            elapsed = time.time() - start

            print(f"\r  A: {answer[:120]}{'...' if len(answer) > 120 else ''}")
            print(f"\n  Judge Score & Reasoning:\n{judge_evaluation}\n")
            print(f"  Time: {elapsed:.1f}s")
            print("-" * 70)

            results.append({
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "chatbot_answer": answer,
                "judge_evaluation": judge_evaluation,
                "elapsed_seconds": round(elapsed, 2),
                "status": "ok",
            })

        except Exception as e:
            print(f"\r  ERROR: {e}")
            results.append({
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "chatbot_answer": None,
                "judge_evaluation": None,
                "error": str(e),
                "status": "error",
            })

    # Save results
    os.makedirs("data", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{'=' * 70}")
    print(f"  Completed: {ok}/{len(GOLDEN_SET)} questions evaluated.")
    print(f"  Results saved to: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    run_judge_evaluation()
