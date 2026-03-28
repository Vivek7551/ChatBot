"""
extract.py
----------
Extracts text from PDFs using marker-pdf to convert to markdown,
dynamically extracts metadata using an LLM, and chunks using 
scispaCy sentence segmentation.
"""

import json
import os
import subprocess
from pathlib import Path
import tiktoken
import spacy
import openai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PDF_DIR = Path("/Users/mynampativivekreddy/Downloads/Research Papers")
OUTPUT_FILE = Path("data/chunks.json")
MARKER_OUT_DIR = Path("data/marker_output")
METADATA_CACHE = Path("data/metadata_cache.json")

CHUNK_SIZE = 500  # Semantic tokens per chunk target

# ---------------------------------------------------------------------------
# Metadata Extraction (Dynamic LLM)
# ---------------------------------------------------------------------------
def extract_metadata_with_llm(raw_text: str, filename: str) -> dict:
    """Uses gpt-4o-mini to read the first page of the paper and extract metadata."""
    
    # 1. Check if we already extracted metadata for this file to save time/API costs
    cache = {}
    if METADATA_CACHE.exists():
        with open(METADATA_CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)
            
    if filename in cache:
        return cache[filename]
        
    print(f"      Calling LLM to extract metadata for {filename}...")
    
    # 2. Setup OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("      [WARN] OPENAI_API_KEY not found. Using fallback metadata.")
        return {"title": filename, "authors": ["Unknown"], "year": 0, "journal": "Unknown", "topics": []}
        
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
        
        # 4. Save to cache so we don't have to call the LLM again for this file
        cache[filename] = meta
        os.makedirs("data", exist_ok=True)
        with open(METADATA_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
            
        return meta
        
    except Exception as e:
        print(f"      [ERROR] LLM Metadata extraction failed: {e}")
        return {"title": filename, "authors": ["Unknown"], "year": 0, "journal": "Unknown", "topics": []}


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------
def extract_pdf_marker(pdf_path: Path) -> str:
    """Uses marker_single CLI to convert PDF to markdown."""
    os.makedirs(MARKER_OUT_DIR, exist_ok=True)
    folder_name = pdf_path.stem
    md_file = MARKER_OUT_DIR / folder_name / f"{folder_name}.md"
    
    # If already extracted, skip to save time
    if md_file.exists():
        with open(md_file, "r", encoding="utf-8") as f:
            return f.read()
    
    cmd = [
        "marker_single",
        str(pdf_path),
        "--output_dir",
        str(MARKER_OUT_DIR)
    ]
    print(f"      Running Marker OCR/Extraction (this may take a minute)...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    if md_file.exists():
        with open(md_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        print(f"      WARNING: Marker failed to output markdown for {pdf_path.name}")
        print(f"      Marker Error Output:\n{res.stderr}")
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

def build_chunks() -> list[dict]:
    """
    Extract, clean, and chunk all PDFs.
    Returns a list of chunk dicts ready for the vector store.
    """
    all_chunks = []
    chunk_id = 0
    encoder = get_encoder()
    nlp = load_scispacy_model()
    
    for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
        filename = pdf_file.name

        print(f"  Extracting: {filename}...")
        raw_text = extract_pdf_marker(pdf_file)
        
        if not raw_text.strip():
            continue
            
        # Dynamically extract metadata
        meta = extract_metadata_with_llm(raw_text, filename)
        
        # Safe-fallbacks just in case the LLM formatting is weird
        title = meta.get("title", filename)
        authors = meta.get("authors", ["Unknown"])
        year = meta.get("year", 0)
        journal = meta.get("journal", "Unknown")
        topics = meta.get("topics", [])
        
        print(f"      ↳ {title[:60]} ({year})")
            
        print(f"      Chunking with scispaCy (target {CHUNK_SIZE} tokens/chunk)...")
        chunks = chunk_text_scispacy(raw_text, nlp, encoder, CHUNK_SIZE)
        
        for i, chunk_text_content in enumerate(chunks):
            if len(chunk_text_content) < 50:
                continue
                
            all_chunks.append({
                "id": chunk_id,
                "text": chunk_text_content,
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "title": title,
                "authors": authors,
                "year": year,
                "journal": journal,
                "topics": topics,
            })
            chunk_id += 1
            
        print(f"    → {len(chunks)} semantic chunks produced\n")

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