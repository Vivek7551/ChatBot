"""
graph_loader.py
---------------
Loads triples from data/triples.json into Neo4j Aura and exposes
GraphRetriever: graph-first chunk retrieval with Chroma dense fallback,
used by rag_engine.py when Neo4j is available.

Usage (one-time, to populate the graph):
    python graph_loader.py

────────────────────────────────────────────────────────────
  NEO4J AURA SETUP (do this once)
────────────────────────────────────────────────────────────
  1. Go to  https://console.neo4j.io
  2. Create a free AuraDB instance (no credit card needed,
     up to 50 000 nodes / 175 000 relationships).
  3. When the instance is created, download the credentials
     .txt file — it contains your URI, username, and password.
  4. Export these three environment variables before running:

     export NEO4J_URI="neo4j+s://<your-instance-id>.databases.neo4j.io"
     export NEO4J_USER="neo4j"
     export NEO4J_PASSWORD="<your-password>"

  The URI scheme MUST be  neo4j+s://  (TLS, certificate verified).
  Using  bolt://  will fail against Aura.
────────────────────────────────────────────────────────────
"""

import json
import os
from pathlib import Path

from neo4j import GraphDatabase, exceptions as neo4j_exc
from dotenv import load_dotenv

# Load local .env if present so Neo4j credentials can be kept in project config.
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRIPLES_PATH = Path("data/triples.json")
CHUNKS_PATH  = Path("data/chunks.json")

# No hardcoded fallbacks — credentials must be provided via environment variables.
NEO4J_URI      = os.environ.get("NEO4J_URI")
NEO4J_USER     = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

# GraphRetriever settings
MAX_HOPS           = 2    # neighbourhood depth for graph traversal
TOP_K_GRAPH        = 15   # max chunk IDs to return from graph traversal
GRAPH_TRAVERSE_LIMIT = 200  # max neighbour nodes to visit per entity (prevents slow unbounded traversals)

# When graph returns at least this many chunks, skip Chroma top-up (still use Chroma if graph is weaker)
_GRAPH_FULL_POOL = int(os.environ.get("GRAPH_PRIMARY_MIN_K", "12"))

# Stopwords to exclude from entity matching — prevents common words
# flooding the full-text index with noise hits.
_STOPWORDS = {
    "what", "which", "where", "when", "does", "show", "cell", "cells",
    "this", "that", "with", "from", "have", "been", "they", "their",
    "also", "into", "both", "such", "each", "used", "using", "were",
    "than", "then", "data", "role", "type", "upon", "able", "made",
    "found", "these", "those", "over", "through", "after", "before",
    "between", "during", "under", "about", "against", "study", "paper",
    "result", "results", "effect", "effects", "level", "levels",
}


# ---------------------------------------------------------------------------
# Driver helper
# ---------------------------------------------------------------------------
def get_driver():
    if not NEO4J_URI:
        raise EnvironmentError(
            "NEO4J_URI is not set.\n"
            "Export it as:  export NEO4J_URI='neo4j+s://<id>.databases.neo4j.io'\n"
            "Get the URI from your AuraDB instance at https://console.neo4j.io"
        )
    if not NEO4J_PASSWORD:
        raise EnvironmentError(
            "NEO4J_PASSWORD is not set.\n"
            "Export it as:  export NEO4J_PASSWORD='<your-password>'\n"
            "The password was shown once when you created the Aura instance.\n"
            "If you lost it, reset it from the Aura console."
        )
    # neo4j+s:// handles TLS and certificate verification automatically
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ---------------------------------------------------------------------------
# Schema setup
# ---------------------------------------------------------------------------
SCHEMA_QUERIES = [
    # Uniqueness constraint — also creates a backing lookup index
    """
    CREATE CONSTRAINT entity_unique IF NOT EXISTS
    FOR (e:Entity) REQUIRE (e.name, e.node_type) IS UNIQUE
    """,
    # Full-text index for fuzzy entity name lookup from query strings
    """
    CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS
    FOR (e:Entity) ON EACH [e.name]
    """,
]

def setup_schema(driver) -> None:
    with driver.session() as session:
        for q in SCHEMA_QUERIES:
            try:
                session.run(q)
            except neo4j_exc.ClientError as e:
                # Constraint/index already exists on re-runs — safe to ignore
                if "already exists" not in str(e).lower():
                    raise
    print("  Schema constraints and indexes ready.")


# ---------------------------------------------------------------------------
# Batch loader
# ---------------------------------------------------------------------------
UPSERT_ENTITY = """
MERGE (e:Entity {name: $name, node_type: $node_type})
ON CREATE SET
    e.chunk_ids = [$chunk_id],
    e.papers    = [$paper_title],
    e.years     = [$year]
ON MATCH SET
    e.chunk_ids = CASE
        WHEN $chunk_id IN e.chunk_ids THEN e.chunk_ids
        ELSE e.chunk_ids + [$chunk_id]
    END,
    e.papers = CASE
        WHEN $paper_title IN e.papers THEN e.papers
        ELSE e.papers + [$paper_title]
    END,
    e.years = CASE
        WHEN $year IN e.years THEN e.years
        ELSE e.years + [$year]
    END
"""

UPSERT_RELATION = """
MATCH (s:Entity {name: $subj_name, node_type: $subj_type})
MATCH (o:Entity {name: $obj_name,  node_type: $obj_type})
MERGE (s)-[r:REL {type: $relation}]->(o)
ON CREATE SET
    r.chunk_ids = [$chunk_id],
    r.evidence  = [$evidence],
    r.count     = 1
ON MATCH SET
    r.chunk_ids = CASE
        WHEN $chunk_id IN r.chunk_ids THEN r.chunk_ids
        ELSE r.chunk_ids + [$chunk_id]
    END,
    r.evidence = CASE
        WHEN $evidence IN r.evidence THEN r.evidence
        ELSE r.evidence + [$evidence]
    END,
    r.count = r.count + 1
"""


def load_triples(driver, triples: list[dict]) -> None:
    """Push all triples to Aura using batched write transactions."""
    total = len(triples)
    print(f"  Loading {total} triples into Neo4j Aura...")

    batch_size = 200
    loaded = 0

    with driver.session() as session:
        for i in range(0, total, batch_size):
            batch = triples[i : i + batch_size]

            with session.begin_transaction() as tx:
                for t in batch:
                    evidence = t.get("evidence", "")[:500]

                    tx.run(UPSERT_ENTITY, {
                        "name":        t["subject"],
                        "node_type":   t["subject_type"],
                        "chunk_id":    t["chunk_id"],
                        "paper_title": t["paper_title"],
                        "year":        t["paper_year"],
                    })
                    tx.run(UPSERT_ENTITY, {
                        "name":        t["object"],
                        "node_type":   t["object_type"],
                        "chunk_id":    t["chunk_id"],
                        "paper_title": t["paper_title"],
                        "year":        t["paper_year"],
                    })
                    tx.run(UPSERT_RELATION, {
                        "subj_name": t["subject"],
                        "subj_type": t["subject_type"],
                        "obj_name":  t["object"],
                        "obj_type":  t["object_type"],
                        "relation":  t["relation"],
                        "chunk_id":  t["chunk_id"],
                        "evidence":  evidence,
                    })
                tx.commit()

            loaded += len(batch)
            print(f"    {loaded}/{total} triples committed...")

    print(f"  Done. {total} triples loaded into Aura.")


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------
def print_graph_stats(driver) -> None:
    with driver.session() as session:
        n_nodes = session.run(
            "MATCH (e:Entity) RETURN count(e) AS c"
        ).single()["c"]

        n_rels = session.run(
            "MATCH ()-[r:REL]->() RETURN count(r) AS c"
        ).single()["c"]

        print(f"\n  Graph statistics:")
        print(f"    Nodes (entities) : {n_nodes}")
        print(f"    Edges (relations): {n_rels}")

        print("\n  Top 10 entities by connection degree:")
        rows = session.run("""
            MATCH (e:Entity)-[r:REL]-()
            RETURN e.name AS name, e.node_type AS type, count(r) AS degree
            ORDER BY degree DESC LIMIT 10
        """)
        for r in rows:
            print(f"    {r['name']:<30} [{r['type']}]  degree={r['degree']}")

        print("\n  Relation type distribution:")
        rows = session.run("""
            MATCH ()-[r:REL]->()
            RETURN r.type AS rel, count(r) AS cnt
            ORDER BY cnt DESC
        """)
        for r in rows:
            print(f"    {r['rel']:<30} {r['cnt']}")


# ---------------------------------------------------------------------------
# GraphRetriever — used by rag_engine.py
# ---------------------------------------------------------------------------
class GraphRetriever:
    """
    Graph-first retrieval against Neo4j Aura:

      1. Entity linking — query tokens → full-text index on Entity nodes
      2. Graph traversal — collect chunk_ids from k-hop neighbourhood (primary)
      3. ChromaDB dense search — fills remaining slots when the graph pool is thin (secondary)
      4. Merge — graph-tagged chunks first; CrossEncoder reranking happens in rag_engine
    """

    def __init__(self):
        self.driver = None

    def connect(self) -> None:
        self.driver = get_driver()
        self.driver.verify_connectivity()
        print("  GraphRetriever connected to Neo4j Aura.")

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    # ------------------------------------------------------------------
    # Entity matching: find graph nodes mentioned in the query
    # ------------------------------------------------------------------
    # Lucene special characters that break query parsing when present in tokens.
    # The most dangerous is '/' which Lucene treats as a regex delimiter.
    _LUCENE_SPECIAL = str.maketrans("", "", r"/+\-&|!(){}[]^\"~*?:")

    def _sanitize_token(self, token: str) -> str:
        """Strip Lucene special chars from a single query token."""
        return token.translate(self._LUCENE_SPECIAL).strip()

    def _extract_query_entities(self, query: str) -> list[str]:
        """
        Uses the full-text index on Aura to find entity names matching
        terms in the query. Stopwords and very short tokens are filtered
        before querying to reduce noise.

        Tokens are sanitized to remove Lucene special characters (e.g. the '/'
        in 'Ras/PI3K') that would cause a TokenMgrError in the Lucene parser.
        """
        raw_tokens = [
            w.strip(".,;:?!()'\"")
            for w in query.split()
            if len(w.strip(".,;:?!'\"")) > 3
        ]
        # Sanitize each token to remove Lucene-special chars
        tokens = [self._sanitize_token(t) for t in raw_tokens]
        # Re-filter: sanitization may have shortened tokens or left them empty
        meaningful_tokens = [
            t for t in tokens
            if len(t) > 2 and t.lower() not in _STOPWORDS
        ]

        if not meaningful_tokens:
            return []

        # Lucene fuzzy query: ~ suffix enables fuzzy matching per term
        lucene_q = " OR ".join(f"{t}~" for t in meaningful_tokens)

        try:
            with self.driver.session() as session:
                rows = session.run(
                    """
                    CALL db.index.fulltext.queryNodes('entity_name_ft', $q)
                    YIELD node, score
                    WHERE score > 0.4
                    RETURN node.name AS name
                    ORDER BY score DESC LIMIT 10
                    """,
                    {"q": lucene_q},
                )
                return [r["name"] for r in rows]
        except Exception as e:
            # Lucene parse errors or transient Neo4j errors — fall back gracefully
            print(f"  [Graph] Fulltext query failed (Chroma dense fallback): {e}")
            return []

    # ------------------------------------------------------------------
    # Graph traversal — collect chunk IDs from the neighbourhood
    # ------------------------------------------------------------------
    def _get_related_chunk_ids(
        self, entity_names: list[str], hops: int = MAX_HOPS
    ) -> list[int]:
        """
        Traverses up to `hops` relationship hops from each matched entity.
        Collects chunk_ids from every node in the neighbourhood.

        A LIMIT clause caps neighbour expansion to avoid slow traversals on
        highly-connected nodes in dense graphs.
        """
        if not entity_names:
            return []

        chunk_id_set: set[int] = set()

        with self.driver.session() as session:
            for name in entity_names:
                rows = session.run(
                    f"""
                    MATCH (start:Entity {{name: $name}})
                    MATCH (start)-[*0..{hops}]-(neighbor:Entity)
                    WITH DISTINCT neighbor LIMIT {GRAPH_TRAVERSE_LIMIT}
                    RETURN neighbor.chunk_ids AS cids
                    """,
                    {"name": name},
                )
                for r in rows:
                    if r["cids"]:
                        chunk_id_set.update(int(c) for c in r["cids"])

        return list(chunk_id_set)

    # ------------------------------------------------------------------
    # Multi-hop path explanation (useful for debugging + /sources command)
    # ------------------------------------------------------------------
    def explain_path(self, entity_a: str, entity_b: str) -> list[str]:
        """
        Returns human-readable steps of the shortest path between two
        entities. Useful for understanding how a multi-hop answer was reached.
        """
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH p = shortestPath(
                    (a:Entity {name: $a})-[*..4]-(b:Entity {name: $b})
                )
                RETURN p LIMIT 1
                """,
                {"a": entity_a, "b": entity_b},
            )
            record = rows.single()
            if not record:
                return [f"No path found between '{entity_a}' and '{entity_b}'"]

            path  = record["p"]
            nodes = list(path.nodes)
            rels  = list(path.relationships)
            steps = []
            for i, rel in enumerate(rels):
                steps.append(
                    f"{nodes[i]['name']} -[{rel['type']}]-> {nodes[i+1]['name']}"
                )
            return steps

    # ------------------------------------------------------------------
    def _score_graph_chunk_ids(
        self,
        query: str,
        vector_store,
        chunk_ids: list[int],
        chunk_lookup: dict,
    ) -> list[dict]:
        """Turn graph-linked chunk IDs into scored chunk dicts (BM25 vs query when available)."""
        if not chunk_ids:
            return []

        from vector_store import tokenize

        tokenized_query = tokenize(query)
        bm25_scores_all = (
            vector_store.bm25.get_scores(tokenized_query)
            if vector_store.bm25 is not None
            else None
        )
        id_list = getattr(vector_store, "_bm25_idx_to_chunk_id", None) or [
            c["id"] for c in vector_store.chunks
        ]
        chunk_id_to_idx = {cid: idx for idx, cid in enumerate(id_list)}

        scored: list[tuple[float, int]] = []
        for cid in chunk_ids:
            if cid not in chunk_lookup:
                continue
            sc = 0.08
            if bm25_scores_all is not None and cid in chunk_id_to_idx:
                raw = float(bm25_scores_all[chunk_id_to_idx[cid]])
                sc = round(raw / (raw + 60.0), 4)
            scored.append((sc, cid))

        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for sc, cid in scored:
            out.append({
                **chunk_lookup[cid],
                "score": sc,
                "graph_retrieved": True,
                "retrieval_source": "graph_traversal",
            })
        return out

    # ------------------------------------------------------------------
    # Main search entry point
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        vector_store,
        all_chunks: list[dict],
        k: int = 20,
    ) -> list[dict]:
        """
        Graph-first retrieval, Chroma dense as secondary:

          1. Entity extraction (query -> Neo4j full-text index)
          2. Graph traversal -> chunk IDs (primary)
          3. Score graph chunks (BM25 vs query), cap at *k*
          4. ChromaDB dense search to fill toward *k* when the graph pool is thin
          5. Merge (graph rows first; dedupe by id keeps first = graph)
        """
        chunk_lookup = {c["id"]: c for c in all_chunks}

        entities = self._extract_query_entities(query)
        graph_chunks: list[dict] = []

        if entities:
            print(f"  [Retrieval] Query entities -> graph: {entities[:5]}")
            graph_ids = self._get_related_chunk_ids(entities)
            print(
                f"  [Retrieval] Graph traversal -> {len(graph_ids)} chunk id(s) (primary)"
            )
            graph_chunks = self._score_graph_chunk_ids(
                query, vector_store, graph_ids, chunk_lookup
            )
        else:
            print(
                "  [Retrieval] No graph entity hits -> Chroma dense fallback only."
            )

        graph_chunks = graph_chunks[:k]
        graph_ids_seen = {c["id"] for c in graph_chunks}

        need_fill = k - len(graph_chunks)
        thin = len(graph_chunks) < min(_GRAPH_FULL_POOL, k)
        want = max(need_fill, (min(_GRAPH_FULL_POOL, k) - len(graph_chunks)) if thin else 0)

        fallback: list[dict] = []
        if want > 0:
            fallback = vector_store.search_chroma_dense(
                query, k=want, exclude_ids=graph_ids_seen
            )
            print(
                f"  [Retrieval] Chroma dense (secondary) -> +{len(fallback)} chunk(s)"
            )

        merged: list[dict] = []
        seen: set[int] = set()
        for c in graph_chunks + fallback:
            cid = c["id"]
            if cid in seen:
                continue
            seen.add(cid)
            merged.append(c)

        if len(merged) < k:
            extra = vector_store.search_chroma_dense(
                query,
                k=k - len(merged),
                exclude_ids=seen,
            )
            for c in extra:
                if c["id"] in seen:
                    continue
                seen.add(c["id"])
                merged.append(c)

        return merged[:k]



# ---------------------------------------------------------------------------
# Main — one-time graph population
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Devreotes GraphRAG — Neo4j Aura Loader")
    print("=" * 60)

    if not TRIPLES_PATH.exists():
        raise FileNotFoundError(
            f"Triples file not found at {TRIPLES_PATH}.\n"
            "Run `python graph_extract.py` first."
        )

    with open(TRIPLES_PATH, "r", encoding="utf-8") as f:
        triples = json.load(f)

    print(f"\n  Loaded {len(triples)} triples from {TRIPLES_PATH}")

    print(f"\n  Connecting to Neo4j Aura...")
    print(f"  URI: {NEO4J_URI or '(not set — export NEO4J_URI first)'}")
    driver = get_driver()
    driver.verify_connectivity()
    print("  Connection verified.")

    print("\n  Setting up schema...")
    setup_schema(driver)

    print()
    load_triples(driver, triples)

    print_graph_stats(driver)

    driver.close()

    print("\n" + "=" * 60)
    print("  Graph loaded into Neo4j Aura successfully.")
    print()
    print("  Next steps:")
    print("    1. Visit https://console.neo4j.io → Explore")
    print("       to visually browse your knowledge graph.")
    print("    2. Run your chatbot — graph retrieval is automatic.")
    print("    3. Try a multi-hop query:")
    print("       'What does the enzyme that degrades PIP3 also regulate?'")
    print("=" * 60)


if __name__ == "__main__":
    main()
