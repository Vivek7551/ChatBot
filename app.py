"""
app.py
------
Flask API server that wraps RAGEngine for the web frontend.

Usage:
    python app.py

Endpoints:
    POST /api/chat       — send a question, get an answer + sources
    GET  /api/papers     — list all papers in the corpus
    POST /api/reset      — clear conversation history
    GET  /health         — health check
"""

import os
import sys
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# ---------------------------------------------------------------------------
# RAGEngine singleton — load once on startup
# ---------------------------------------------------------------------------
_engine = None
_engine_lock = threading.Lock()
_engine_error = None


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


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON with a 'question' field"}), 400

    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        engine = get_engine()
        answer, chunks = engine.ask(question)

        # Build sources list — use .get() with defaults so missing fields never crash
        seen = {}
        for c in chunks:
            src = c.get("source", "")
            if src not in seen:
                seen[src] = {
                    "title":   c.get("title", "Unknown title"),
                    "year":    c.get("year") or "",
                    "authors": c.get("authors") or [],
                    "score":   round(float(c.get("score") or 0), 4),
                    "journal": c.get("journal") or "",
                }

        return jsonify({
            "answer":  answer,
            "sources": list(seen.values()),
        })

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        import traceback
        traceback.print_exc()   # prints full stack trace to Flask terminal
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/papers", methods=["GET"])
def papers():
    try:
        engine = get_engine()
        paper_list = engine.get_paper_list()
        return jsonify({"papers": paper_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    try:
        engine = get_engine()
        engine.reset_conversation()
        return jsonify({"status": "reset"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n  Devreotes Lab Chatbot API — starting on http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
