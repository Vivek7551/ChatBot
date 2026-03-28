"""
ragas_eval.py
-------------
Evaluates the RAG pipeline using the industry-standard Ragas framework.
Calculates metrics such as Faithfulness, Answer Relevancy, Context Precision,
and Context Recall.

Requires: pip install "ragas>=0.1,<0.2" datasets
"""

import os
import json

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
except ImportError:
    print("ERROR: missing required libraries.")
    print('Please install them by running: pip install "ragas>=0.1,<0.2" datasets')
    exit(1)

from rag_engine import RAGEngine

# User-provided Golden Set
GOLDEN_SET = [
    {
        "question": "Is cAR1 phosphorylation required for chemotaxis?",
        "ground_truth": "No. Elimination of phosphorylation via site-directed substitution or C-terminal truncation does not impair chemotaxis, aggregation, or the adaptation of adenylyl cyclase."
    },
    {
        "question": "How does adenosine affect cAMP adaptation?",
        "ground_truth": "Adenosine blocks the adaptation of the cAMP signaling response. If adenosine is removed while cAMP is present, cells respond immediately, proving they had not adapted."
    },
    {
        "question": "Does folic acid trigger adaptation like cAMP?",
        "ground_truth": "No. Folic acid elicits minimal or no adaptation. Pretreatment with folic acid does not impair the cell's subsequent response to cAMP."
    },
    {
        "question": "What are the three modules of the 2011 chemotaxis model?",
        "ground_truth": "The model consists of: 1) LEGI (gradient sensing/adaptation), 2) EN (excitable network/motility), and 3) POL (polarization/persistence)."
    }
]

def run_ragas_evaluation():
    print("=" * 70)
    print("  Devreotes RAG Chatbot — Ragas Evaluation")
    print("=" * 70)

    engine = RAGEngine()
    engine.load()

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    print("\n[1/2] Answering Golden Set queries via RAG Engine...")
    for i, item in enumerate(GOLDEN_SET, 1):
        print(f"  -> Processing Q{i}: {item['question']}")
        engine.reset_conversation()

        answer, chunks = engine.ask(item["question"], multi_turn=False)
        contexts = [chunk["text"] for chunk in chunks]

        data["question"].append(item["question"])
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(item["ground_truth"])

    print("\n[2/2] Running Ragas Evaluation Metrics...")
    print("This will send the generated responses, contexts, and ground truths to the LLM for scoring.")

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    print("\n" + "=" * 70)
    print("  Ragas Evaluation Summary")
    print("=" * 70)
    print(result)

    os.makedirs("data", exist_ok=True)
    df = result.to_pandas()

    output_file = "data/ragas_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed metric breakdown per question saved to: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    run_ragas_evaluation()
