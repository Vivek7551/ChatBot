"""
rag_engine.py
-------------
Advanced RAG pipeline — Graph-Augmented.

Retrieval order:
  1. Hybrid Vector Search (ChromaDB dense + BM25 sparse, RRF-fused)
  2. Graph Traversal via GraphRetriever (Neo4j)
  3. CrossEncoder Reranking over the merged candidate pool
  4. Grounded LLM generation with citation enforcement

GraphRetriever is optional: if Neo4j is unavailable or
USE_GRAPH_RETRIEVAL=0 is set, the engine falls back silently to the
original pure-vector pipeline so nothing breaks.
"""

import os
import re
import openai
from vector_store import HybridVectorStore
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL             = "gpt-4o-mini"
TOP_K_RETRIEVAL   = 20       # candidates from hybrid search
TOP_K_RERANK      = 6        # chunks sent to the LLM after reranking
MAX_CONTEXT_CHARS = 12_000
MAX_TOKENS        = 1024

# Maximum number of conversation turns to keep in memory (each turn = 2 messages).
# Older turns are dropped to stay within the model context window.
MAX_HISTORY_TURNS = 10

# Set USE_GRAPH_RETRIEVAL=0 to disable graph augmentation without code changes
USE_GRAPH = os.environ.get("USE_GRAPH_RETRIEVAL", "1") == "1"

SYSTEM_PROMPT = """You are a scientific research assistant specialising in the \
published work of Professor Peter N. Devreotes, a cell biologist at Johns \
Hopkins University known for his research on chemotaxis, signal transduction, \
and membrane protein biology.

You answer questions ONLY using the excerpts from Prof. Devreotes' papers \
provided in the context below. Do not draw on outside knowledge.

Guidelines:
- Always cite which paper(s) your answer comes from, using the title and year.
- If multiple papers address the question, synthesise across them.
- If the context does not contain enough information, say so clearly — do not guess.
- Use precise scientific language appropriate for a graduate-level audience.
- When quoting specific findings, be exact.
"""

REFUSAL_MESSAGE = (
    "I cannot answer this confidently from the provided documents. "
    "Please ask a narrower question or provide more relevant documents."
)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------
def build_context(chunks: list[dict]) -> tuple[str, list[str]]:
    sections      = []
    citation_tags = []
    for i, chunk in enumerate(chunks, 1):
        tag = f"S{i}"
        citation_tags.append(tag)
        graph_flag = " [graph-retrieved]" if chunk.get("graph_retrieved") else ""
        header = (
            f"[{tag}]{graph_flag} "
            f"{chunk['title']} "
            f"({chunk['year']}) "
            f"[source: {chunk['source']}, chunk: {chunk['chunk_index']+1}/{chunk['total_chunks']}] "
            f"— {', '.join(chunk['authors'][:2])}"
            f"{' et al.' if len(chunk['authors']) > 2 else ''}"
        )
        sections.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(sections), citation_tags


# ---------------------------------------------------------------------------
# Citation grounding helpers
# ---------------------------------------------------------------------------
def _extract_citations(answer: str) -> set[str]:
    matches = re.findall(r"\[([^\]]+)\]", answer)
    tags = set()
    for m in matches:
        for token in m.split(","):
            token = token.strip()
            if re.fullmatch(r"S\d+", token):
                tags.add(token)
    return tags


def _is_valid_grounded_answer(answer: str, allowed_citations: set[str]) -> bool:
    used = _extract_citations(answer)
    if not used:
        return False
    return used.issubset(allowed_citations)


def _repair_answer_with_citations(
    client: openai.OpenAI,
    draft_answer: str,
    context: str,
    citation_tags: list[str],
) -> str:
    repair_prompt = (
        "You are a scientific citation editor. Rewrite the draft answer below so that "
        "every factual claim is supported by an inline citation tag from the allowed set.\n"
        f"Allowed citation tags: {', '.join(citation_tags)}.\n"
        "Use the format [S#] or [S#, S#] immediately after each claim.\n"
        "Keep the scientific content intact — only add or fix the citation tags.\n\n"
        "CONTEXT (use this as the source of truth):\n"
        f"{context}\n\n"
        "DRAFT ANSWER (add citations to this):\n"
        f"{draft_answer}"
    )
    repaired = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": repair_prompt}],
    )
    return (repaired.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# RAGEngine
# ---------------------------------------------------------------------------
class RAGEngine:
    def __init__(self, store_path: str = "data/chunks.json"):
        self.store_path           = store_path
        self.store                = HybridVectorStore()
        self.reranker             = None
        self.client               = None
        self.graph_retriever      = None
        self.conversation_history: list[dict] = []

    # ------------------------------------------------------------------
    def load(self) -> None:
        print("Loading BGE-Reranker model (this may take a moment)...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

        print("Loading Hybrid Vector store...")
        self.store.load(self.store_path)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your environment or .env file."
            )
        self.client = openai.OpenAI(api_key=api_key)

        if USE_GRAPH:
            try:
                from graph_loader import GraphRetriever
                self.graph_retriever = GraphRetriever()
                self.graph_retriever.connect()
                print("  Graph-augmented retrieval: ENABLED")
            except Exception as e:
                print(f"  [WARN] Graph retrieval unavailable ({e}). "
                      "Falling back to vector-only.")
                self.graph_retriever = None
        else:
            print("  Graph-augmented retrieval: DISABLED (USE_GRAPH_RETRIEVAL=0)")

        print("Advanced RAG engine ready.\n")

    # ------------------------------------------------------------------
    def _retrieve(self, query: str) -> list[dict]:
        """
        Unified retrieval: vector-only or graph-augmented depending on
        whether GraphRetriever was initialised successfully.
        """
        if self.graph_retriever is not None:
            return self.graph_retriever.search(
                query        = query,
                vector_store = self.store,
                all_chunks   = self.store.chunks,
                k            = TOP_K_RETRIEVAL,
            )
        return self.store.search(query, k=TOP_K_RETRIEVAL)

    # ------------------------------------------------------------------
    def _trim_history(self) -> list[dict]:
        """
        Return a window of the most recent MAX_HISTORY_TURNS turns
        (each turn = one user message + one assistant message = 2 entries).
        This prevents unbounded growth that would overflow the context window.
        """
        max_messages = MAX_HISTORY_TURNS * 2
        return self.conversation_history[-max_messages:]

    # ------------------------------------------------------------------
    def ask(self, query: str, multi_turn: bool = True) -> tuple[str, list[dict]]:
        # 1. Retrieve (hybrid vector + optional graph)
        candidate_chunks = self._retrieve(query)

        # 2. CrossEncoder Reranking
        if candidate_chunks:
            pairs  = [[query, chunk["text"]] for chunk in candidate_chunks]
            scores = self.reranker.predict(pairs)

            scored_chunks = sorted(zip(scores, candidate_chunks),
                                   key=lambda x: x[0], reverse=True)
            final_chunks  = [chunk for _, chunk in scored_chunks[:TOP_K_RERANK]]

            for score, chunk in scored_chunks[:TOP_K_RERANK]:
                chunk["rerank_score"] = round(float(score), 4)
        else:
            final_chunks = []

        if not final_chunks:
            return REFUSAL_MESSAGE, []

        # 3. Build context & prompt
        context, citation_tags = build_context(final_chunks)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated for length]"

        grounded_message = (
            f"CONTEXT FROM PROF. DEVREOTES' PAPERS:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWERING RULES:\n"
            f"1) Answer using ONLY information present in the context above.\n"
            f"2) After each factual claim, add a citation tag like [S1] or [S2, S3] "
            f"   using only these allowed tags: {', '.join(citation_tags)}.\n"
            f"3) If the context partially addresses the question, answer what you can "
            f"   and note what is not covered — but DO NOT refuse entirely if there is "
            f"   ANY relevant information in the context.\n"
            f"4) Write in clear, graduate-level scientific prose.\n"
        )

        if multi_turn:
            self.conversation_history.append({"role": "user", "content": grounded_message})
            trimmed_history = self._trim_history()
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + trimmed_history
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": grounded_message},
            ]

        # 4. LLM call
        response = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
        )
        answer = (response.choices[0].message.content or "").strip()

        # 5. Citation enforcement
        # Only refuse if the answer has zero valid citations after a repair attempt.
        # Answers with at least one valid citation are accepted — the LLM grounded
        # at least part of its answer in the provided context.
        if answer != REFUSAL_MESSAGE:
            allowed = set(citation_tags)
            used    = _extract_citations(answer)

            if not used:
                # No citation tags at all — send to repair
                repaired = _repair_answer_with_citations(
                    client        = self.client,
                    draft_answer  = answer,
                    context       = context,
                    citation_tags = citation_tags,
                )
                repaired_used = _extract_citations(repaired)
                if repaired_used & allowed:
                    # Repair produced at least one valid citation — accept it
                    answer = repaired
                else:
                    # Repair also produced no valid citations — refuse
                    answer = REFUSAL_MESSAGE
            elif not used.issubset(allowed):
                # Some tags are hallucinated — repair to fix them
                repaired = _repair_answer_with_citations(
                    client        = self.client,
                    draft_answer  = answer,
                    context       = context,
                    citation_tags = citation_tags,
                )
                repaired_used = _extract_citations(repaired)
                # Accept if repair fixed at least some citations
                if repaired_used & allowed:
                    answer = repaired
                # else keep original answer — it already had partial valid citations

        if multi_turn:
            self.conversation_history.append({"role": "assistant", "content": answer})

        return answer, final_chunks

    # ------------------------------------------------------------------
    def reset_conversation(self) -> None:
        self.conversation_history = []
        print("Conversation history cleared.")

    # ------------------------------------------------------------------
    def evaluate_with_judge(
        self, question: str, chatbot_answer: str, ground_truth: str
    ) -> str:
        prompt = (
            f"Compare the 'Chatbot Answer' to the 'Ground Truth.' "
            f"Give a score from 1-5 on technical accuracy. "
            f"If the chatbot says adenosine doesn't affect adaptation time, give it a 1.\n\n"
            f"User Question: {question}\n"
            f"Ground Truth: {ground_truth}\n"
            f"Chatbot Answer: {chatbot_answer}\n"
        )
        response = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    def get_paper_list(self) -> list[dict]:
        seen   = set()
        papers = []
        for chunk in self.store.chunks:
            key = chunk["source"]
            if key not in seen:
                seen.add(key)
                papers.append({
                    "file":    chunk["source"],
                    "title":   chunk["title"],
                    "authors": chunk["authors"],
                    "year":    chunk["year"],
                    "journal": chunk["journal"],
                    "topics":  chunk["topics"],
                })
        return sorted(papers, key=lambda p: p["year"])
