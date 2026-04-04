"""
paper_ingest.py
---------------
Incremental pipeline for a single new PDF: extract → chunk → vector append →
triple extraction → merge triples.json → Neo4j MERGE (author + new bio only).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import openai

from extract import METADATA_CACHE, build_chunks_for_pdf
from graph_extract import (
    TRIPLES_PATH,
    extract_biology_triples_from_chunks,
    generate_author_triples_for_sources,
    merge_triples_after_ingest,
    _load_scispacy,
)
from graph_loader import load_triples
from ontology_validator import OntologyAnnotator

if TYPE_CHECKING:
    from vector_store import HybridVectorStore

CHUNKS_PATH = Path("data/chunks.json")


def _next_chunk_id(chunks: list) -> int:
    if not chunks:
        return 0
    return max(int(c["id"]) for c in chunks) + 1


def incremental_ingest_pdf(
    pdf_path: Path,
    store: "HybridVectorStore",
    engine_lock: Optional[threading.Lock] = None,
    *,
    log: bool = True,
) -> dict:
    """
    Process one PDF already saved on disk. Thread-safe if *engine_lock* is provided
    for all store reads/writes.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return {"ok": False, "error": "Not a PDF file."}

    def locked(run):
        if engine_lock:
            with engine_lock:
                return run()
        return run()

    def read_existing():
        return list(store.chunks)

    existing = locked(read_existing)
    if any(c.get("source") == pdf_path.name for c in existing):
        return {
            "ok": False,
            "error": f"The corpus already includes a file named “{pdf_path.name}”.",
        }

    start_id = _next_chunk_id(existing)
    meta_cache: dict = {}
    if METADATA_CACHE.exists():
        with open(METADATA_CACHE, "r", encoding="utf-8") as f:
            meta_cache.update(json.load(f))

    if log:
        print(f"  [ingest] Extracting & chunking {pdf_path.name}…")
    new_chunks, meta_cache = build_chunks_for_pdf(pdf_path, start_id, meta_cache)

    if not new_chunks:
        return {
            "ok": False,
            "error": "No text could be extracted (Marker/OCR failed or PDF is empty).",
        }

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"ok": False, "error": "OPENAI_API_KEY is not set."}

    merged = existing + new_chunks

    if log:
        print(f"  [ingest] Knowledge-graph extraction ({len(new_chunks)} chunks)…")
    client = openai.OpenAI(api_key=api_key)
    annotator = OntologyAnnotator()
    nlp = _load_scispacy()
    new_bio = extract_biology_triples_from_chunks(
        new_chunks,
        client=client,
        annotator=annotator,
        nlp=nlp,
        sleep_s=0.2,
        log_prefix="  [ingest]" if log else "",
    )

    merge_triples_after_ingest(merged, new_chunks, new_bio, TRIPLES_PATH)

    author_delta = generate_author_triples_for_sources(merged, {pdf_path.name})
    neo4j_msg = None
    try:
        from graph_loader import get_driver

        driver = get_driver()
        load_triples(driver, author_delta + new_bio)
        driver.close()
        if log:
            print("  [ingest] Neo4j updated.")
    except Exception as e:
        neo4j_msg = str(e)
        if log:
            print(f"  [ingest] Neo4j skipped: {e}")

    def append_and_save():
        store.append_chunks(new_chunks)
        os.makedirs(CHUNKS_PATH.parent, exist_ok=True)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(store.chunks, f, ensure_ascii=False, indent=2)

    locked(append_and_save)

    return {
        "ok": True,
        "source": pdf_path.name,
        "chunks_added": len(new_chunks),
        "bio_triples": len(new_bio),
        "neo4j_note": neo4j_msg,
    }
