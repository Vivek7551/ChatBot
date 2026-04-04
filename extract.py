"""
extract.py
----------
Extracts text from PDFs using marker-pdf to convert to markdown,
dynamically extracts metadata using an LLM, and chunks using 
scispaCy sentence segmentation.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import tiktoken
import openai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_PDF_DIR = _REPO_ROOT / "papers"
PDF_DIR = Path(os.environ.get("GRAPH_RAG_PDF_DIR", str(_DEFAULT_PDF_DIR)))
OUTPUT_FILE = Path("data/chunks.json")
MARKER_OUT_DIR = Path("data/marker_output")
METADATA_CACHE = Path("data/metadata_cache.json")

CHUNK_SIZE = 500  # Semantic tokens per chunk target

# ---------------------------------------------------------------------------
# Metadata Extraction (Dynamic LLM)
# ---------------------------------------------------------------------------
def _normalize_metadata(meta: dict, filename: str) -> dict:
    """Coerce LLM JSON into stable types for downstream chunking and graph code."""
    title = meta.get("title") or filename
    if not isinstance(title, str):
        title = str(title)

    authors = meta.get("authors", ["Unknown"])
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.replace(" and ", ", ").split(",") if a.strip()]
    elif not isinstance(authors, list):
        authors = ["Unknown"]
    else:
        authors = [str(a).strip() for a in authors if str(a).strip()]
    if not authors:
        authors = ["Unknown"]

    year = meta.get("year", 0)
    try:
        year = int(year)
    except (TypeError, ValueError):
        year = 0

    journal = meta.get("journal", "Unknown")
    if not isinstance(journal, str):
        journal = str(journal) if journal else "Unknown"

    topics = meta.get("topics", [])
    if isinstance(topics, str):
        topics = [topics] if topics else []
    elif not isinstance(topics, list):
        topics = []
    else:
        topics = [str(t).strip() for t in topics if str(t).strip()]

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "journal": journal,
        "topics": topics,
    }


def _save_metadata_cache(cache: dict) -> None:
    os.makedirs(METADATA_CACHE.parent, exist_ok=True)
    with open(METADATA_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def extract_metadata_with_llm(
    raw_text: str, filename: str, cache: Optional[dict] = None
) -> tuple[dict, dict]:
    """
    Uses gpt-4o-mini to read the start of the paper and extract metadata.
    Pass a shared *cache* dict (loaded once per run); it is updated in place when
    new metadata is fetched. Returns (metadata, cache).
    """
    if cache is None:
        cache = {}
        if METADATA_CACHE.exists():
            with open(METADATA_CACHE, "r", encoding="utf-8") as f:
                cache.update(json.load(f))

    if filename in cache:
        return _normalize_metadata(cache[filename], filename), cache
        
    print(f"      Calling LLM to extract metadata for {filename}...")
    
    # 2. Setup OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("      [WARN] OPENAI_API_KEY not found. Using fallback metadata.")
        return _normalize_metadata({}, filename), cache
        
    client = openai.OpenAI(api_key=api_key)
    
    # 3. Take just the beginning of the text (abstract/header) to save tokens
    text_snippet = raw_text[:4000]
    
    prompt = f"""
    Analyze the following excerpt from an academic research paper and extract the core metadata.
    You must output a valid JSON object with EXACTLY these keys:
    - "title" (string: The full title of the paper)
    - "authors" (list of strings: The names of the authors)
    - "year" (integer: The year of publication. If you cannot find it, return 0)
    - "journal" (string: The journal it was published in. If unknown, return "Unknown")
    - "topics" (list of strings: 3 to 5 biological keywords or topics)

    Text Excerpt:
    {text_snippet}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={ "type": "json_object" }
        )
        
        meta = json.loads(response.choices[0].message.content)
        meta = _normalize_metadata(meta, filename)

        cache[filename] = meta
        _save_metadata_cache(cache)

        return meta, cache

    except Exception as e:
        print(f"      [ERROR] LLM Metadata extraction failed: {e}")
        fb = _normalize_metadata({}, filename)
        return fb, cache


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------
def _clean_marker_markdown(text: str) -> str:
    """Normalise line endings and collapse runaway blank lines from PDF conversion."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_marker(pdf_path: Path) -> str:
    """Uses marker_single CLI to convert PDF to markdown."""
    os.makedirs(MARKER_OUT_DIR, exist_ok=True)
    folder_name = pdf_path.stem
    md_file = MARKER_OUT_DIR / folder_name / f"{folder_name}.md"

    if md_file.exists():
        with open(md_file, "r", encoding="utf-8") as f:
            return _clean_marker_markdown(f.read())

    cmd = [
        "marker_single",
        str(pdf_path),
        "--output_dir",
        str(MARKER_OUT_DIR),
    ]
    print("      Running Marker OCR/Extraction (this may take a minute)...")
    res = subprocess.run(cmd, capture_output=True, text=True)

    if md_file.exists():
        with open(md_file, "r", encoding="utf-8") as f:
            return _clean_marker_markdown(f.read())

    if res.returncode != 0:
        print(f"      WARNING: marker_single exited with code {res.returncode} for {pdf_path.name}")
    print(f"      WARNING: Marker failed to output markdown for {pdf_path.name}")
    err = (res.stderr or res.stdout or "").strip()
    if err:
        print(f"      Marker output:\n{err[:2000]}")
    return ""


def get_encoder():
    try:
        return tiktoken.encoding_for_model("gpt-4o")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def load_scispacy_model():
    """
    Load a scientific spaCy model via direct import to bypass symlink issues.
    """
    try:
        import en_core_sci_sm
        return en_core_sci_sm.load()
    except ImportError as e:
        raise RuntimeError(
            "scispaCy model not found in this specific Python environment.\n"
            "Run this exact command:\n"
            "python -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.5/en_core_sci_sm-0.5.5.tar.gz"
        ) from e


def chunk_text_scispacy(text: str, nlp, encoder, chunk_size: int) -> list[str]:
    """
    Build chunks by grouping scispaCy-detected sentences up to chunk_size tokens.
    """
    doc = nlp(text)
    chunks = []
    current_sentences = []
    current_tokens = 0

    for sent in doc.sents:
        sentence = sent.text.strip()
        if not sentence:
            continue
        sent_tokens = len(encoder.encode(sentence))

        # Flush current chunk before adding an overflowing sentence.
        if current_sentences and current_tokens + sent_tokens > chunk_size:
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_tokens = 0

        # If a single sentence is huge, split by token slices.
        if sent_tokens > chunk_size:
            token_ids = encoder.encode(sentence)
            for i in range(0, len(token_ids), chunk_size):
                piece = encoder.decode(token_ids[i:i + chunk_size]).strip()
                if piece:
                    chunks.append(piece)
            continue

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def build_chunks_for_pdf(
    pdf_path: Path,
    start_chunk_id: int,
    meta_cache: Optional[dict],
    nlp=None,
    encoder=None,
) -> tuple[list[dict], dict]:
    """
    Extract, metadata, and chunk a single PDF. Returns (chunk dicts, updated meta_cache).
    *start_chunk_id* is the next global chunk id to assign.
    Pass shared *nlp* / *encoder* when batching many PDFs to avoid reloading models.
    """
    if meta_cache is None:
        meta_cache = {}
        if METADATA_CACHE.exists():
            with open(METADATA_CACHE, "r", encoding="utf-8") as f:
                meta_cache.update(json.load(f))

    if nlp is None:
        nlp = load_scispacy_model()
    if encoder is None:
        encoder = get_encoder()

    filename = pdf_path.name
    raw_text = extract_pdf_marker(pdf_path)
    if not raw_text.strip():
        return [], meta_cache

    meta, meta_cache = extract_metadata_with_llm(raw_text, filename, meta_cache)
    title, authors, year, journal, topics = (
        meta["title"],
        meta["authors"],
        meta["year"],
        meta["journal"],
        meta["topics"],
    )

    raw_chunks = chunk_text_scispacy(raw_text, nlp, encoder, CHUNK_SIZE)
    kept = [c for c in raw_chunks if len(c) >= 50]
    total_kept = len(kept)

    out: list[dict] = []
    cid = start_chunk_id
    for i, chunk_text_content in enumerate(kept):
        out.append({
            "id": cid,
            "text": chunk_text_content,
            "source": filename,
            "chunk_index": i,
            "total_chunks": total_kept,
            "title": title,
            "authors": authors,
            "year": year,
            "journal": journal,
            "topics": topics,
        })
        cid += 1

    return out, meta_cache


def build_chunks() -> list[dict]:
    """
    Extract, clean, and chunk all PDFs.
    Returns a list of chunk dicts ready for the vector store.
    """
    all_chunks = []
    chunk_id = 0
    encoder = get_encoder()
    nlp = load_scispacy_model()
    meta_cache: dict = {}
    if METADATA_CACHE.exists():
        with open(METADATA_CACHE, "r", encoding="utf-8") as f:
            meta_cache.update(json.load(f))

    if not PDF_DIR.is_dir():
        print(f"  [WARN] PDF directory does not exist: {PDF_DIR}")
        print("         Add PDFs under ./papers/ or set GRAPH_RAG_PDF_DIR.")
        return []

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"  [WARN] No PDF files found in {PDF_DIR}")
        return []

    for pdf_file in pdfs:
        filename = pdf_file.name
        print(f"  Extracting: {filename}...")
        added, meta_cache = build_chunks_for_pdf(
            pdf_file, chunk_id, meta_cache, nlp=nlp, encoder=encoder
        )
        if not added:
            continue
        print(f"      ↳ {added[0]['title'][:60]} ({added[0]['year']})")
        print(f"      Chunking with scispaCy (target {CHUNK_SIZE} tokens/chunk)...")
        print(f"    → {len(added)} semantic chunks kept\n")
        all_chunks.extend(added)
        chunk_id += len(added)

    return all_chunks

def main():
    os.makedirs("data", exist_ok=True)
    print(f"\nExtracting text from PDFs in {PDF_DIR} using marker-pdf...")
    chunks = build_chunks()
    print(f"\nTotal chunks: {len(chunks)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Saved to {OUTPUT_FILE}")

    if chunks:
        print(f"\nSample chunk from first paper:")
        print("-" * 60)
        print(chunks[0]["text"][:300])
        print("-" * 60)

if __name__ == "__main__":
    main()