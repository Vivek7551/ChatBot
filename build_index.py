"""
build_index.py
--------------
One-time script: extracts text from all PDFs and builds the Hybrid Vector index.
Run this whenever you add new papers to the corpus.

    python build_index.py
"""

import time
from extract import build_chunks
from vector_store import HybridVectorStore

def main():
    start = time.time()
    print("=" * 60)
    print("  Devreotes GraphRAG — Hybrid Index Builder")
    print("=" * 60)

    # Step 1: Extract and chunk all PDFs (Semantic Chunking & Marker)
    print("\n[1/2] Extracting and chunking PDFs...")
    chunks = build_chunks()
    print(f"      Total semantic chunks produced: {len(chunks)}")

    # Step 2: Build Hybrid Vector Store (ChromaDB + BM25)
    print("\n[2/2] Building Hybrid Dense/Sparse vector index...")
    store = HybridVectorStore()
    store.build(chunks)
    store.save("data/chunks.json")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Ready to chat — run: python chatbot.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
