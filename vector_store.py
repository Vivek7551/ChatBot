"""
vector_store.py
---------------
Hybrid Vector Store combining ChromaDB (Dense) and BM25Okapi (Sparse)
via Reciprocal Rank Fusion (RRF).

Dense embeddings use OpenAI's 'text-embedding-3-large'.
"""

import json
import os
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI
import re
from dotenv import load_dotenv

load_dotenv()


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, return word tokens for BM25."""
    text = text.lower()
    return re.findall(r"\b[a-z][a-z0-9\-]{1,}\b", text)


class HybridVectorStore:

    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="devreotes_papers")
        self.bm25 = None
        self.chunks: list[dict] = []
        # Explicit mapping: BM25 corpus index → chunk["id"]
        # This ensures BM25 rank positions are never confused with chunk IDs,
        # which is only safe when IDs are contiguous integers starting at 0.
        self._bm25_idx_to_chunk_id: list[int] = []
        self._built = False

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your environment or .env file."
            )
        self.client = OpenAI(api_key=api_key)

    def build(self, chunks: list[dict]) -> None:
        """Build the Hybrid index."""
        self.chunks = chunks
        print(f"  Building Hybrid vector store over {len(chunks)} chunks...")

        if not chunks:
            print("  WARNING: 0 chunks provided. Skipping vector DB build.")
            self._built = True
            return

        try:
            self.chroma_client.delete_collection("devreotes_papers")
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection("devreotes_papers")

        texts     = [chunk["text"] for chunk in chunks]
        ids       = [str(chunk["id"]) for chunk in chunks]
        metadatas = [
            {k: str(v) if isinstance(v, list) else v for k, v in chunk.items() if k != "text"}
            for chunk in chunks
        ]

        print("  Generating OpenAI embeddings (text-embedding-3-large)...")
        embeddings = []
        for i in range(0, len(texts), 200):
            batch = texts[i:i + 200]
            res = self.client.embeddings.create(input=batch, model="text-embedding-3-large")
            embeddings.extend([r.embedding for r in res.data])

        print("  Adding to ChromaDB...")
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        print("  Building Sparse BM25 Index...")
        tokenized_corpus = [tokenize(doc) for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        # Record the chunk ID that corresponds to each BM25 corpus position
        self._bm25_idx_to_chunk_id = [chunk["id"] for chunk in chunks]
        self._built = True
        print("  Hybrid Index built.")

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Return top-k chunks using Hybrid Search (ChromaDB + BM25 fused via RRF).
        """
        if not self._built:
            raise RuntimeError("Call build() or load() before search().")

        res_embed = self.client.embeddings.create(input=[query], model="text-embedding-3-large")
        query_emb = res_embed.data[0].embedding

        TOP_K_INITIAL = max(k * 3, 20)

        # 1. Chroma Query
        chroma_res = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(TOP_K_INITIAL, len(self.chunks))
        )
        dense_ids = [int(i) for i in chroma_res["ids"][0]]

        # 2. BM25 Query
        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:TOP_K_INITIAL]

        # 3. RRF Fusion
        k_rf = 60
        rrf_scores: dict[int, float] = {}

        for rank, chunk_id in enumerate(dense_ids):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rank + k_rf)

        # Use the explicit index map so BM25 corpus positions are correctly
        # translated to chunk IDs regardless of how IDs were assigned.
        for rank, corpus_idx in enumerate(top_bm25_indices):
            chunk_id = self._bm25_idx_to_chunk_id[int(corpus_idx)]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rank + k_rf)

        sorted_rrf  = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_fused = sorted_rrf[:k]

        chunk_lookup = {c["id"]: c for c in self.chunks}
        results = []
        for chunk_id, score in top_k_fused:
            if chunk_id not in chunk_lookup:
                continue
            result = dict(chunk_lookup[chunk_id])
            result["score"] = round(score, 4)
            results.append(result)

        return results

    def save(self, path: str = "data/chunks.json") -> None:
        """
        Since ChromaDB is persistent on disk, we only need to save the chunks
        JSON; BM25 is rebuilt quickly on load.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        print(f"  Saved document chunks to {path} (ChromaDB saved to data/chroma_db/)")

    def load(self, path: str = "data/chunks.json") -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Chunks not found at {path}. Run `python build_index.py` first."
            )
        print(f"  Loading chunks from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print("  Building BM25 in-memory index from chunks...")
        tokenized_corpus = [tokenize(doc["text"]) for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        # Rebuild the index map so search() translates positions correctly
        self._bm25_idx_to_chunk_id = [chunk["id"] for chunk in self.chunks]
        self._built = True
        print(f"  Loaded {len(self.chunks)} chunks. Vector DB ready.")
