"""
graph_extract.py
----------------
Reads chunks.json and extracts biological entity-relation-entity triples
using a 3-stage pipeline:

  Stage 0 — Author triples: deterministic, zero cost, from chunk metadata
  Stage 1 — Entity detection: BioPortal ontology + scispaCy NER fusion
  Stage 2 — Triple extraction: OpenAI LLM with strong prompt + retry
  Stage 3 — Post-filter: confidence scoring, deduplication

Features:
  • Resume support (GRAPH_RESUME=1)
  • Progress display with per-chunk stats
  • Retry on rate-limit / timeout errors
  • Author/paper graph built automatically from metadata
  • Compatible with graph_loader.py (paper_title, source provenance)

Usage:
    python graph_extract.py

Environment variables:
    OPENAI_API_KEY      — required
    BIOPORTAL_API_KEY   — optional; enables ontology grounding
    GRAPH_MODEL         — model override (default: gpt-4o-mini)
    GRAPH_RESUME        — set to "1" to resume from previous run
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import openai
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from ontology_validator import OntologyAnnotator, OntologyHit

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL        = os.environ.get("GRAPH_MODEL", "gpt-4o-mini")
CHUNKS_PATH  = Path("data/chunks.json")
TRIPLES_PATH = Path("data/triples.json")
SLEEP_BETWEEN = 0.3   # seconds between API calls

# ---------------------------------------------------------------------------
# Node & relation type registries
# ---------------------------------------------------------------------------
ALLOWED_NODE_TYPES = {
    "Protein", "Lipid", "Gene", "SmallMolecule", "Receptor",
    "Process", "Pathway", "CellType", "Organism", "Structure",
    "Phenotype", "Disease", "Method", "Model", "Condition",
    "BiophysicalProperty", "EngineeredConstruct", "BiologicalEntity",
    # Author graph (deterministic — not LLM-extracted)
    "Author", "Paper",
}

ALLOWED_RELATION_TYPES = {
    # Biochemical — HIGH priority
    "PHOSPHORYLATES", "DEPHOSPHORYLATES", "BINDS",
    "ACTIVATES", "INHIBITS", "PRODUCES", "DEGRADES",
    # Spatial
    "LOCALIZES_TO", "RECRUITS_TO", "ENRICHED_AT", "EXCLUDED_FROM",
    # Genetic / causal
    "ENCODES", "REGULATES", "CAUSES", "RESCUES",
    # Functional
    "REQUIRED_FOR", "SUFFICIENT_FOR", "MEDIATES",
    "USED_TO_STUDY", "ASSOCIATED_WITH",
    # Model / meta
    "SIMULATES", "PREDICTS", "PUBLISHED_IN",
    "OCCURS_DURING", "MODULATES", "REPORTS_ON",
    "CONSUMES", "GENERATES",
    # Author graph (deterministic)
    "AUTHORED_BY", "FIRST_AUTHOR_OF", "COLLABORATED_WITH",
}

# Author-graph relations skip LLM validation — they are always valid
_AUTHOR_RELATIONS = {"AUTHORED_BY", "FIRST_AUTHOR_OF", "COLLABORATED_WITH"}

# High-confidence biochemical relations used for confidence scoring
_HIGH_CONFIDENCE_RELATIONS = {
    "PHOSPHORYLATES", "DEPHOSPHORYLATES", "BINDS",
    "ACTIVATES", "INHIBITS", "PRODUCES", "DEGRADES",
}

# ---------------------------------------------------------------------------
# scispaCy loader (optional — graceful fallback if not installed)
# ---------------------------------------------------------------------------
def _load_scispacy():
    try:
        import en_core_sci_sm
        return en_core_sci_sm.load()
    except Exception:
        logger.warning("scispaCy model not available — NER disabled. "
                       "Install with: pip install en_core_sci_sm")
        return None


def _extract_ner_entities(text: str, nlp) -> list[str]:
    """Return unique entity strings found by scispaCy NER."""
    if nlp is None:
        return []
    doc = nlp(text)
    return list({ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 2})


# ---------------------------------------------------------------------------
# Entity context builder (BioPortal + NER fusion)
# ---------------------------------------------------------------------------
def _build_entity_context(ontology_hits: list[OntologyHit], ner_entities: list[str]) -> str:
    lines = []
    for h in ontology_hits:
        canonical = h.pref_label if h.pref_label != h.term else h.term
        lines.append(f"  - {h.term}"
                     + (f" (canonical: {canonical})" if canonical != h.term else "")
                     + f"  [type={h.node_type}, source=ontology:{h.ontology}]")
    # Add NER entities not already covered by ontology
    ont_terms_lower = {h.term.lower() for h in ontology_hits}
    for e in ner_entities:
        if e.lower() not in ont_terms_lower:
            lines.append(f"  - {e}  [type=BiologicalEntity, source=scispacy_ner]")
    return "\n".join(lines) if lines else "  (none detected)"


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a biomedical knowledge-graph extraction system specialising
in the research of Prof. Peter Devreotes (chemotaxis, signal transduction, PI3K/PTEN
signalling, actin dynamics, membrane biology).

Your task: extract entity–relation–entity triples from the text.

RULES:
1. Only extract relationships EXPLICITLY stated in the text. No inference.
2. Prioritise entities listed in the ENTITY HINTS section — use their canonical names.
3. Use only the ALLOWED RELATION TYPES listed below.
4. Assign confidence: "high" for strong biochemical verbs, "medium" for causal/regulatory,
   "low" for vague associations.
5. Never extract a triple where subject == object.
6. Reply in valid JSON only — no prose.

ALLOWED RELATION TYPES (in priority order):
HIGH:   PHOSPHORYLATES, DEPHOSPHORYLATES, BINDS, ACTIVATES, INHIBITS, PRODUCES, DEGRADES
MEDIUM: LOCALIZES_TO, RECRUITS_TO, ENRICHED_AT, EXCLUDED_FROM,
        ENCODES, REGULATES, CAUSES, RESCUES,
        REQUIRED_FOR, SUFFICIENT_FOR, MEDIATES, USED_TO_STUDY
LOW:    MODULATES, OCCURS_DURING, GENERATES, CONSUMES, ASSOCIATED_WITH
SKIP:   SIMULATES, PREDICTS, PUBLISHED_IN, REPORTS_ON (use these rarely)
"""


def _build_prompt(chunk: dict, entity_context: str) -> str:
    return f"""Paper: {chunk['title']} ({chunk['year']}) — {chunk['journal']}

ENTITY HINTS (prefer these as nodes, use canonical names shown):
{entity_context}

TEXT:
{chunk['text']}

Return JSON:
{{
  "triples": [
    {{
      "subject":      "<entity name>",
      "subject_type": "<NodeType>",
      "relation":     "<RELATION_TYPE>",
      "object":       "<entity name>",
      "object_type":  "<NodeType>",
      "evidence":     "<verbatim excerpt supporting this triple>",
      "confidence":   "high|medium|low"
    }}
  ]
}}
If no triples, return: {{"triples": []}}
"""


# ---------------------------------------------------------------------------
# OpenAI call with retry
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
    ),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_llm(client: openai.OpenAI, chunk: dict, entity_context: str) -> list[dict]:
    """Call the LLM and return raw triple dicts."""
    prompt = _build_prompt(chunk, entity_context)
    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    try:
        return json.loads(content).get("triples", [])
    except json.JSONDecodeError:
        logger.warning(f"Bad JSON on chunk {chunk['id']}: {content[:120]}")
        return []


# ---------------------------------------------------------------------------
# Triple validation & filtering
# ---------------------------------------------------------------------------
def _is_valid_triple(t: dict) -> bool:
    """Return True if a triple passes all quality gates."""
    required = {"subject", "subject_type", "relation", "object", "object_type"}
    if not required.issubset(t.keys()):
        return False
    subj = (t["subject"] or "").strip()
    obj  = (t["object"]  or "").strip()
    if not subj or not obj or subj == obj:
        return False
    if len(subj) < 2 or len(obj) < 2:
        return False
    # Author triples bypass relation-type check (they're always valid)
    if t["relation"] in _AUTHOR_RELATIONS:
        return True
    if t["relation"] not in ALLOWED_RELATION_TYPES:
        return False
    # Skip low-confidence ASSOCIATED_WITH to reduce noise
    if t["relation"] == "ASSOCIATED_WITH" and t.get("confidence") == "low":
        return False
    return True


def _coerce_node_type(label: Optional[str]) -> str:
    """Map LLM output to an allowed Neo4j node label; unknown → BiologicalEntity."""
    t = (label or "").strip()
    if t in ALLOWED_NODE_TYPES:
        return t
    return "BiologicalEntity"


def _enrich_triple(t: dict, chunk: dict) -> dict:
    """Add provenance fields that graph_loader.py expects."""
    t["subject_type"] = _coerce_node_type(t.get("subject_type"))
    t["object_type"] = _coerce_node_type(t.get("object_type"))
    t["chunk_id"]    = chunk["id"]
    t["source"]      = chunk["source"]      # PDF filename
    t["paper_title"] = chunk["title"]       # graph_loader.py reads this
    t["paper_year"]  = chunk["year"]
    t.setdefault("ontology_source", None)
    t.setdefault("confidence", "medium")
    # Upgrade confidence for known high-value relations (after relation is validated upstream)
    if t["relation"] in _HIGH_CONFIDENCE_RELATIONS and t.get("confidence") == "medium":
        t["confidence"] = "high"
    return t


# ---------------------------------------------------------------------------
# Author triple generation — deterministic, zero API cost
# ---------------------------------------------------------------------------
def generate_author_triples(chunks: list[dict]) -> list[dict]:
    """
    Build Author ↔ Paper triples from chunk metadata (no LLM needed).

    Enables queries like:
      "What papers did Devreotes publish?"
      "Who collaborated most with Peter Devreotes?"
      "List all papers where Devreotes was first author."

    Relations:
      Paper  -[AUTHORED_BY]->       Author   (all authors)
      Author -[FIRST_AUTHOR_OF]->   Paper    (first author only)
      Author -[COLLABORATED_WITH]-> Author   (all co-author pairs, bidirectional)
    """
    seen_papers: dict[str, dict] = {}
    for chunk in chunks:
        seen_papers.setdefault(chunk["source"], chunk)

    triples: list[dict] = []

    for paper_chunk in seen_papers.values():
        title   = paper_chunk["title"]
        year    = paper_chunk["year"]
        src     = paper_chunk["source"]
        cid     = paper_chunk["id"]
        authors = paper_chunk.get("authors", [])
        if not authors:
            continue

        # Paper → AUTHORED_BY → Author
        for author in authors:
            triples.append({
                "subject": title,        "subject_type": "Paper",
                "relation": "AUTHORED_BY",
                "object":  author,       "object_type":  "Author",
                "evidence": f"{title} ({year}) authored by {author}",
                "confidence": "high",    "ontology_source": None,
                "chunk_id": cid,         "source": src,
                "paper_title": title,    "paper_year": year,
            })

        # First Author → FIRST_AUTHOR_OF → Paper
        triples.append({
            "subject": authors[0],      "subject_type": "Author",
            "relation": "FIRST_AUTHOR_OF",
            "object":  title,           "object_type":  "Paper",
            "evidence": f"{authors[0]} is first author of {title} ({year})",
            "confidence": "high",       "ontology_source": None,
            "chunk_id": cid,            "source": src,
            "paper_title": title,       "paper_year": year,
        })

        # Author → COLLABORATED_WITH → Author (bidirectional pairs)
        for i, a in enumerate(authors):
            for b in authors[i + 1:]:
                for subj, obj in [(a, b), (b, a)]:
                    triples.append({
                        "subject": subj,       "subject_type": "Author",
                        "relation": "COLLABORATED_WITH",
                        "object":  obj,        "object_type":  "Author",
                        "evidence": f"{subj} and {obj} co-authored {title} ({year})",
                        "confidence": "high",  "ontology_source": None,
                        "chunk_id": cid,       "source": src,
                        "paper_title": title,  "paper_year": year,
                    })

    return triples


def generate_author_triples_for_sources(chunks: list[dict], sources: set[str]) -> list[dict]:
    """Like generate_author_triples but only papers whose PDF filename is in *sources*."""
    if not sources:
        return []
    seen_papers: dict[str, dict] = {}
    for chunk in chunks:
        if chunk["source"] in sources:
            seen_papers.setdefault(chunk["source"], chunk)
    triples: list[dict] = []
    for paper_chunk in seen_papers.values():
        title = paper_chunk["title"]
        year = paper_chunk["year"]
        src = paper_chunk["source"]
        cid = paper_chunk["id"]
        authors = paper_chunk.get("authors", [])
        if not authors:
            continue
        for author in authors:
            triples.append({
                "subject": title,
                "subject_type": "Paper",
                "relation": "AUTHORED_BY",
                "object": author,
                "object_type": "Author",
                "evidence": f"{title} ({year}) authored by {author}",
                "confidence": "high",
                "ontology_source": None,
                "chunk_id": cid,
                "source": src,
                "paper_title": title,
                "paper_year": year,
            })
        triples.append({
            "subject": authors[0],
            "subject_type": "Author",
            "relation": "FIRST_AUTHOR_OF",
            "object": title,
            "object_type": "Paper",
            "evidence": f"{authors[0]} is first author of {title} ({year})",
            "confidence": "high",
            "ontology_source": None,
            "chunk_id": cid,
            "source": src,
            "paper_title": title,
            "paper_year": year,
        })
        for i, a in enumerate(authors):
            for b in authors[i + 1:]:
                for subj, obj in [(a, b), (b, a)]:
                    triples.append({
                        "subject": subj,
                        "subject_type": "Author",
                        "relation": "COLLABORATED_WITH",
                        "object": obj,
                        "object_type": "Author",
                        "evidence": f"{subj} and {obj} co-authored {title} ({year})",
                        "confidence": "high",
                        "ontology_source": None,
                        "chunk_id": cid,
                        "source": src,
                        "paper_title": title,
                        "paper_year": year,
                    })
    return triples


def extract_biology_triples_from_chunks(
    chunks: list[dict],
    *,
    client: openai.OpenAI,
    annotator: OntologyAnnotator,
    nlp,
    sleep_s: float = SLEEP_BETWEEN,
    log_prefix: str = "",
) -> list[dict]:
    """Run ontology + LLM extraction for each chunk; returns validated biological triples."""
    out: list[dict] = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks, 1):
        if log_prefix:
            short_title = (chunk.get("title") or "")[:52]
            print(f"{log_prefix} [{idx}/{total}] chunk {chunk['id']} | {short_title}...", end=" ", flush=True)
        ontology_hits = annotator.annotate(chunk["text"])
        ner_entities = _extract_ner_entities(chunk["text"], nlp)
        entity_ctx = _build_entity_context(ontology_hits, ner_entities)
        raw_triples: list[dict] = []
        try:
            raw_triples = _call_llm(client, chunk, entity_ctx)
        except openai.APIError as e:
            if log_prefix:
                print(f"\n    [WARN] API error chunk {chunk['id']}: {e}")
        except Exception as e:
            if log_prefix:
                print(f"\n    [WARN] Unexpected error chunk {chunk['id']}: {e}")
        valid = []
        for t in raw_triples:
            t = _enrich_triple(t, chunk)
            if _is_valid_triple(t):
                valid.append(t)
        out.extend(valid)
        if log_prefix:
            print(f"→ {len(valid)} triples")
        time.sleep(sleep_s)
    return out


def merge_triples_after_ingest(
    all_chunks: list[dict],
    new_chunks: list[dict],
    new_bio_triples: list[dict],
    triples_path: Path,
) -> list[dict]:
    """
    Rewrite triples.json: fresh author graph for full corpus + old biological triples
    (excluding any chunk_ids we are replacing) + *new_bio_triples*.
    """
    new_ids = {c["id"] for c in new_chunks}
    sources_new = {c["source"] for c in new_chunks}
    existing: list[dict] = []
    if triples_path.exists():
        with open(triples_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    bio_kept = [
        t
        for t in existing
        if t.get("relation") not in _AUTHOR_RELATIONS
        and t.get("chunk_id") not in new_ids
        and t.get("source") not in sources_new
    ]
    author_all = generate_author_triples(all_chunks)
    merged = author_all + bio_kept + new_bio_triples
    os.makedirs(triples_path.parent, exist_ok=True)
    with open(triples_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return merged


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def _load_existing(path: Path) -> tuple[list[dict], set[int]]:
    if not path.exists():
        return [], set()
    with open(path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    processed = {t["chunk_id"] for t in existing if "chunk_id" in t}
    return existing, processed


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs("data", exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks not found at {CHUNKS_PATH}. Run `python build_index.py` first."
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    client    = openai.OpenAI(api_key=api_key)
    annotator = OntologyAnnotator()
    nlp       = _load_scispacy()

    print("=" * 65)
    print("  Devreotes GraphRAG — Triple Extractor")
    print("=" * 65)
    print(f"  Model              : {MODEL}")
    print(f"  Ontology grounding : {'ENABLED' if annotator._available else 'DISABLED (no BIOPORTAL_API_KEY)'}")
    print(f"  scispaCy NER       : {'ENABLED' if nlp else 'DISABLED'}")
    print(f"  Chunks loaded      : {len(chunks)}")

    # Resume support
    resume = os.environ.get("GRAPH_RESUME", "0") == "1"
    if resume:
        all_triples, processed_ids = _load_existing(TRIPLES_PATH)
        to_process = [c for c in chunks if c["id"] not in processed_ids]
        print(f"  Resuming           : {len(processed_ids)} done, {len(to_process)} remaining")
    else:
        all_triples  = []
        processed_ids = set()
        to_process   = chunks

    # ── Stage 0: Author triples (instant, deterministic) ──────────────────
    print("\n  [Stage 0] Generating author/paper triples from metadata...")
    author_triples = generate_author_triples(chunks)
    if not resume:
        all_triples.extend(author_triples)
    else:
        existing_evid = {t.get("evidence", "") for t in all_triples}
        new_auth = [t for t in author_triples if t["evidence"] not in existing_evid]
        all_triples.extend(new_auth)
    n_papers  = len({t["paper_title"] for t in author_triples})
    n_authors = len({t["object"] for t in author_triples if t["relation"] == "AUTHORED_BY"})
    print(f"  → {len(author_triples)} triples | {n_papers} papers | {n_authors} authors\n")

    # ── Stage 1+2+3: LLM extraction per chunk ─────────────────────────────
    total      = len(to_process)
    n_dropped  = 0
    n_high     = 0

    print(f"  [Stage 1-3] Extracting biological triples from {total} chunks...\n")

    for idx, chunk in enumerate(to_process, 1):
        short_title = chunk["title"][:52]
        print(f"  [{idx:>4}/{total}] chunk {chunk['id']:>4} | "
              f"{chunk['year']} | {short_title}...", end=" ", flush=True)

        # Stage 1 — entity detection
        ontology_hits = annotator.annotate(chunk["text"])
        ner_entities  = _extract_ner_entities(chunk["text"], nlp)
        entity_ctx    = _build_entity_context(ontology_hits, ner_entities)

        # Stage 2 — LLM triple extraction
        raw_triples: list[dict] = []
        try:
            raw_triples = _call_llm(client, chunk, entity_ctx)
        except openai.APIError as e:
            print(f"\n    [WARN] API error chunk {chunk['id']}: {e}")
        except Exception as e:
            print(f"\n    [WARN] Unexpected error chunk {chunk['id']}: {e}")

        # Stage 3 — enrich + validate + filter
        valid = []
        for t in raw_triples:
            t = _enrich_triple(t, chunk)
            if _is_valid_triple(t):
                valid.append(t)

        n_dropped += len(raw_triples) - len(valid)
        chunk_high  = sum(1 for t in valid if t.get("confidence") == "high")
        n_high     += chunk_high

        all_triples.extend(valid)
        print(f"→ {len(valid)} triples (high={chunk_high}, "
              f"dropped={len(raw_triples) - len(valid)})")

        # Incremental save (enables resume)
        with open(TRIPLES_PATH, "w", encoding="utf-8") as f:
            json.dump(all_triples, f, ensure_ascii=False, indent=2)

        time.sleep(SLEEP_BETWEEN)

    # Summary (count from merged file so resume runs report correct totals)
    n_author_in_all = sum(1 for t in all_triples if t["relation"] in _AUTHOR_RELATIONS)
    n_bio_in_all = len(all_triples) - n_author_in_all
    print("\n" + "=" * 65)
    print("  Extraction complete.")
    print(f"  Author triples      : {n_author_in_all}")
    print(f"  Biological triples  : {n_bio_in_all}")
    print(f"  High-confidence     : {n_high}")
    print(f"  Dropped (filtered)  : {n_dropped}")
    print(f"  Total saved         : {len(all_triples)}")
    print(f"  Output              : {TRIPLES_PATH}")

    from collections import Counter
    rel_counts = Counter(t["relation"] for t in all_triples)
    print("\n  Top relation types:")
    for rel, cnt in rel_counts.most_common(12):
        print(f"    {rel:<30} {cnt}")

    print("=" * 65)
    print("\n  Next: run `python graph_loader.py` to push to Neo4j.\n")


if __name__ == "__main__":
    main()
