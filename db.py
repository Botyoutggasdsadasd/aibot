"""
SQLite database layer.
Tables:
  users(telegram_id, name, age, school, grade, track, ai_name, created_at)
  messages(id, telegram_id, role, content, created_at)   -- chat history for context + admin review
  ocr_cache(id, telegram_id, extracted_text, created_at) -- last OCR'd content, so buttons can act on it
"""
import sqlite3
import time
from contextlib import contextmanager
from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    telegram_id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    school TEXT,
    grade TEXT,
    track TEXT,
    ai_name TEXT,
    state TEXT DEFAULT 'idle',
    created_at INTEGER
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id INTEGER,
    role TEXT,       -- 'user' or 'assistant'
    content TEXT,
    created_at INTEGER
);

CREATE TABLE IF NOT EXISTS ocr_cache (
    telegram_id INTEGER PRIMARY KEY,
    extracted_text TEXT,
    created_at INTEGER
);
"""

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)

def upsert_user(telegram_id, **fields):
    with get_conn() as conn:
        cur = conn.execute("SELECT telegram_id FROM users WHERE telegram_id=?", (telegram_id,))
        exists = cur.fetchone()
        if exists:
            cols = ", ".join(f"{k}=?" for k in fields)
            conn.execute(f"UPDATE users SET {cols} WHERE telegram_id=?",
                         (*fields.values(), telegram_id))
        else:
            fields["telegram_id"] = telegram_id
            fields["created_at"] = int(time.time())
            cols = ", ".join(fields.keys())
            qs = ", ".join("?" for _ in fields)
            conn.execute(f"INSERT INTO users ({cols}) VALUES ({qs})", tuple(fields.values()))

def get_user(telegram_id):
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM users WHERE telegram_id=?", (telegram_id,))
        row = cur.fetchone()
        return dict(row) if row else None

def set_state(telegram_id, state):
    upsert_user(telegram_id, state=state)

def save_message(telegram_id, role, content):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (telegram_id, role, content, created_at) VALUES (?,?,?,?)",
            (telegram_id, role, content, int(time.time())),
        )

def get_recent_history(telegram_id, limit=12):
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT role, content FROM messages WHERE telegram_id=? ORDER BY id DESC LIMIT ?",
            (telegram_id, limit),
        )
        rows = [dict(r) for r in cur.fetchall()]
        return list(reversed(rows))

def save_ocr(telegram_id, text):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO ocr_cache (telegram_id, extracted_text, created_at) VALUES (?,?,?) "
            "ON CONFLICT(telegram_id) DO UPDATE SET extracted_text=excluded.extracted_text, created_at=excluded.created_at",
            (telegram_id, text, int(time.time())),
        )

def get_ocr(telegram_id):
    with get_conn() as conn:
        cur = conn.execute("SELECT extracted_text FROM ocr_cache WHERE telegram_id=?", (telegram_id,))
        row = cur.fetchone()
        return row["extracted_text"] if row else None

def all_users():
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM users ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]

def user_count():
    with get_conn() as conn:
        cur = conn.execute("SELECT COUNT(*) c FROM users")
        return cur.fetchone()["c"]
