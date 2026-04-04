"""
chat_store.py
-------------
SQLite persistence for multi-chat UI (per-chat message history).
Prepared for future user accounts (add user_id column later).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional

DEFAULT_DB_PATH = Path("data/chat_history.db")


def _conn(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(path))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    with _conn(db_path) as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
            """
        )
        c.commit()


def create_chat(title: str = "New chat", db_path: Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    init_db(db_path)
    cid = str(uuid.uuid4())
    now = time.time()
    with _conn(db_path) as c:
        c.execute(
            "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (cid, title[:200], now, now),
        )
        c.commit()
    return {"id": cid, "title": title[:200], "created_at": now, "updated_at": now}


def list_chats(limit: int = 80, db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    init_db(db_path)
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def update_chat_title(chat_id: str, title: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    init_db(db_path)
    with _conn(db_path) as c:
        c.execute(
            "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
            (title[:200], time.time(), chat_id),
        )
        c.commit()


def touch_chat(chat_id: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    init_db(db_path)
    with _conn(db_path) as c:
        c.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (time.time(), chat_id))
        c.commit()


def delete_chat(chat_id: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    if not db_path.exists():
        return
    init_db(db_path)
    with _conn(db_path) as c:
        c.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        c.commit()


def append_message(
    chat_id: str,
    role: str,
    content: str,
    sources: Optional[list] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    init_db(db_path)
    now = time.time()
    src = json.dumps(sources) if sources is not None else None
    with _conn(db_path) as c:
        cur = c.execute(
            "INSERT INTO messages (chat_id, role, content, sources_json, created_at) VALUES (?,?,?,?,?)",
            (chat_id, role, content, src, now),
        )
        c.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
        c.commit()
        return int(cur.lastrowid)


def get_messages(chat_id: str, db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    init_db(db_path)
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT id, role, content, sources_json, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("sources_json"):
            try:
                d["sources"] = json.loads(d["sources_json"])
            except json.JSONDecodeError:
                d["sources"] = []
        else:
            d["sources"] = None
        del d["sources_json"]
        out.append(d)
    return out


def get_llm_history(chat_id: str, db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, str]]:
    """OpenAI-style messages for RAG (user + assistant text only)."""
    msgs = get_messages(chat_id, db_path)
    hist = []
    for m in msgs:
        if m["role"] not in ("user", "assistant"):
            continue
        hist.append({"role": m["role"], "content": m["content"]})
    return hist
