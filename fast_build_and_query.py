import json
from vector_store import HybridVectorStore
from rag_engine import RAGEngine
import os

def check():
    if not os.path.exists("data/chunks.json"):
        print("chunks.json missing. Cannot run fast evaluation.")
        return

    with open("data/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks.")
    
    # 1. Rebuild index from existing chunks to skip extraction waiting
    print("Building Hybrid Index for evaluation...")
    store = HybridVectorStore()
    store.build(chunks)

    # 2. Setup RAG Engine
    engine = RAGEngine()
    engine.load()
    engine.store = store

    # 3. Query
    q = "According to the 1983 and 1984 papers, how do folic acid and adenosine differ in their effect on the cAMP adaptation mechanism?"
    print(f"\nQUERY: {q}\n")
    ans, retrieved = engine.ask(q, multi_turn=False)
    
    print("\n" + "="*50)
    print("RAG ANSWER:")
    print("="*50)
    print(ans)
    print("="*50)

if __name__ == "__main__":
    check()
