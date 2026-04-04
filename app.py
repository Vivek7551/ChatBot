"""
app.py
------
Flask API for the Devreotes Lab chat UI: multi-chat history, conversational RAG,
and incremental PDF ingestion.

Usage:
    python app.py
"""

import os
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from dotenv import load_dotenv

import chat_store
from chat_store import DEFAULT_DB_PATH
from paper_ingest import incremental_ingest_pdf

load_dotenv()

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024  # 80 MB PDFs

PAPERS_DIR = Path(os.environ.get("GRAPH_RAG_PDF_DIR", str(Path(__file__).resolve().parent / "papers")))

# ---------------------------------------------------------------------------
# Engine singleton + locks
# ---------------------------------------------------------------------------
_engine = None
_engine_lock = threading.Lock()
_engine_error = None

ingest_lock = threading.Lock()
ingest_status: dict = {
    "state": "idle",
    "message": "",
    "detail": None,
}


def get_engine():
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    if _engine_error is not None:
        raise RuntimeError(_engine_error)
    with _engine_lock:
        if _engine is None:
            try:
                from rag_engine import RAGEngine

                engine = RAGEngine()
                engine.load()
                _engine = engine
            except Exception as e:
                _engine_error = str(e)
                raise
    return _engine


chat_store.init_db(DEFAULT_DB_PATH)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory("frontend", "favicon.ico", mimetype="image/x-icon")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# --- Chats -----------------------------------------------------------------


@app.route("/api/chats", methods=["GET"])
def list_chats():
    return jsonify({"chats": chat_store.list_chats(db_path=DEFAULT_DB_PATH)})


@app.route("/api/chats", methods=["POST"])
def create_chat():
    data = request.get_json(force=True, silent=True) or {}
    title = (data.get("title") or "New chat").strip() or "New chat"
    row = chat_store.create_chat(title=title, db_path=DEFAULT_DB_PATH)
    return jsonify(row), 201


@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def remove_chat(chat_id):
    chat_store.delete_chat(chat_id, db_path=DEFAULT_DB_PATH)
    return jsonify({"status": "deleted"})


@app.route("/api/chats/<chat_id>/messages", methods=["GET"])
def get_messages(chat_id):
    ids = {c["id"] for c in chat_store.list_chats(db_path=DEFAULT_DB_PATH)}
    if chat_id not in ids:
        return jsonify({"error": "Chat not found"}), 404
    return jsonify({"messages": chat_store.get_messages(chat_id, db_path=DEFAULT_DB_PATH)})


def _sources_from_chunks(chunks: list) -> list:
    seen = {}
    for c in chunks:
        src = c.get("source", "")
        if src not in seen:
            seen[src] = {
                "title": c.get("title", "Unknown title"),
                "year": c.get("year") or "",
                "authors": c.get("authors") or [],
                "score": round(float(c.get("score") or 0), 4),
                "journal": c.get("journal") or "",
            }
    return list(seen.values())


@app.route("/api/chats/<chat_id>/messages", methods=["POST"])
def send_chat_message(chat_id):
    chats = {c["id"] for c in chat_store.list_chats(db_path=DEFAULT_DB_PATH)}
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404

    data = request.get_json(force=True, silent=True) or {}
    content = (data.get("content") or "").strip()
    if not content:
        return jsonify({"error": "content is required"}), 400

    prior = chat_store.get_llm_history(chat_id, db_path=DEFAULT_DB_PATH)

    try:
        engine = get_engine()
        with _engine_lock:
            answer, chunks = engine.ask_with_history(content, prior)
        sources = _sources_from_chunks(chunks)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    chat_store.append_message(chat_id, "user", content, db_path=DEFAULT_DB_PATH)
    chat_store.append_message(
        chat_id, "assistant", answer, sources=sources, db_path=DEFAULT_DB_PATH
    )

    if len(prior) == 0:
        title = content[:72] + ("…" if len(content) > 72 else "")
        chat_store.update_chat_title(chat_id, title, db_path=DEFAULT_DB_PATH)

    return jsonify({"answer": answer, "sources": sources})


# --- Legacy single-session chat (optional) --------------------------------


@app.route("/api/chat", methods=["POST"])
def chat_legacy():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        engine = get_engine()
        with _engine_lock:
            answer, chunks = engine.ask(question, multi_turn=True)
        return jsonify({"answer": answer, "sources": _sources_from_chunks(chunks)})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/papers", methods=["GET"])
def papers():
    try:
        engine = get_engine()
        with _engine_lock:
            paper_list = engine.get_paper_list()
        return jsonify({"papers": paper_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    try:
        engine = get_engine()
        with _engine_lock:
            engine.reset_conversation()
        return jsonify({"status": "reset"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Ingest ---------------------------------------------------------------

ingest_thread: threading.Thread | None = None


def _ingest_worker(saved: Path):
    global ingest_status
    try:
        with ingest_lock:
            ingest_status = {
                "state": "running",
                "message": "Extracting text and updating indexes…",
                "detail": None,
            }
        engine = get_engine()
        result = incremental_ingest_pdf(
            saved, engine.store, engine_lock=_engine_lock, log=True
        )
        if not result.get("ok"):
            with ingest_lock:
                ingest_status = {
                    "state": "error",
                    "message": result.get("error", "Ingest failed"),
                    "detail": result,
                }
            if "already includes" in str(result.get("error", "")):
                saved.unlink(missing_ok=True)
            return
        with ingest_lock:
            ingest_status = {
                "state": "done",
                "message": "Paper indexed and graph updated.",
                "detail": result,
            }
    except Exception as e:
        import traceback

        traceback.print_exc()
        with ingest_lock:
            ingest_status = {"state": "error", "message": str(e), "detail": None}


@app.route("/api/papers/upload", methods=["POST"])
def upload_paper():
    global ingest_thread, ingest_status
    if "file" not in request.files:
        return jsonify({"error": "Missing file field (expected name: file)"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    name = secure_filename(f.filename)
    if not name.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF uploads are supported"}), 400

    with ingest_lock:
        if ingest_status.get("state") == "running":
            return jsonify({"error": "Another paper is still being processed."}), 409
        ingest_status = {"state": "queued", "message": "Queued…", "detail": None}

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    dest = PAPERS_DIR / name
    if dest.exists():
        with ingest_lock:
            ingest_status = {"state": "idle", "message": "", "detail": None}
        return jsonify(
            {
                "error": f"A file named {name} already exists on disk. "
                "Remove it or rename before uploading."
            }
        ), 409

    f.save(str(dest))

    t = threading.Thread(target=_ingest_worker, args=(dest,), daemon=True)
    ingest_thread = t
    t.start()
    return jsonify({"ok": True, "filename": name, "status": "started"})


@app.route("/api/ingest/status", methods=["GET"])
def ingest_state():
    with ingest_lock:
        return jsonify(dict(ingest_status))


# --- Auth placeholders ------------------------------------------------------


@app.route("/api/auth/signin", methods=["POST"])
def auth_signin_placeholder():
    return jsonify(
        {
            "ok": False,
            "message": "Sign-in is not enabled yet. Chat history is stored locally on the server.",
        }
    ), 501


@app.route("/api/auth/signup", methods=["POST"])
def auth_signup_placeholder():
    return jsonify(
        {
            "ok": False,
            "message": "Sign-up is not enabled yet.",
        }
    ), 501


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n  Devreotes Lab Chatbot API — http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
