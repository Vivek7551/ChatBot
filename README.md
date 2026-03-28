# 🧬 Devreotes Lab Research Chatbot

A graph-augmented RAG (Retrieval-Augmented Generation) chatbot for exploring Prof. Peter Devreotes' research on chemotaxis, signal transduction, PI3K/PTEN signalling, and membrane biology.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Flask](https://img.shields.io/badge/Flask-3.x-green) ![Neo4j](https://img.shields.io/badge/Neo4j-Aura-teal) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange)

---

## ✨ Features

- **Hybrid search** — ChromaDB dense vectors + BM25 sparse search, RRF-fused
- **Graph-augmented retrieval** — Neo4j knowledge graph with ontology-grounded entities (BioPortal, GO, ChEBI)
- **Author/paper graph** — query who collaborated with whom, list papers by author
- **CrossEncoder reranking** — BGE-Reranker-base for precision retrieval
- **Premium web UI** — dark-mode chat interface with citation badges, paper sidebar, and sources panel
- **Conversation history** — multi-turn Q&A with context memory

---

## 🛠️ Prerequisites

- Python 3.12+
- A [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/) free instance
- An [OpenAI API key](https://platform.openai.com/api-keys)
- _(Optional)_ A [BioPortal API key](https://bioportal.bioontology.org/account) for ontology grounding

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Vivek7551/ChatBot.git
cd ChatBot
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up environment variables

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...

NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

BIOPORTAL_API_KEY=your-key-here   # optional
```

### 4. Add your PDFs and build the index

Place your PDF files in a `pdfs/` folder, then run:

```bash
python extract.py        # Extract text from PDFs
python build_index.py    # Build the vector index (chunks.json + ChromaDB)
```

### 5. Extract the knowledge graph and load into Neo4j

```bash
python graph_extract.py   # Extract triples (saves to data/triples.json)
python graph_loader.py    # Push triples to Neo4j
```

> **Tip:** If extraction is interrupted, resume with `GRAPH_RESUME=1 python graph_extract.py`

### 6. Start the chatbot

```bash
python app.py
```

Open your browser at **http://localhost:5001** 🎉

---

## 📁 Project Structure

```
ChatBot/
├── app.py                  # Flask API server
├── rag_engine.py           # Core RAG pipeline
├── vector_store.py         # Hybrid ChromaDB + BM25 store
├── graph_loader.py         # Neo4j graph retriever
├── graph_extract.py        # LLM-based triple extraction
├── ontology_validator.py   # BioPortal ontology grounding
├── extract.py              # PDF extraction
├── build_index.py          # Index builder
├── chatbot.py              # Terminal chatbot (CLI mode)
├── frontend/
│   └── index.html          # Web UI
├── data/                   # Generated data (gitignored)
├── .env.example            # Environment variable template
└── requirements.txt
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | `{"question": "..."}` → answer + sources |
| `GET` | `/api/papers` | List all papers in the corpus |
| `POST` | `/api/reset` | Clear conversation history |
| `GET` | `/health` | Health check |

---

## 💡 Example Questions

- _"What is the role of PI3K in chemotaxis?"_
- _"How does PTEN regulate PIP3 at the leading edge?"_
- _"What papers did Devreotes publish?"_
- _"Who collaborated with Peter Devreotes?"_
- _"Describe the actin dynamics during cell polarization."_

---

## 📝 License

MIT
