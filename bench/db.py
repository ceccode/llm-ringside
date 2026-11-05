"""SQLite persistence. One table for runs, one for per-model results."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .config import ROOT

DB_PATH = ROOT / "bench.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  prompt TEXT NOT NULL,
  system TEXT,
  params TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  model_id TEXT NOT NULL,
  rep INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  error TEXT,
  http_status INTEGER,
  request_id TEXT,
  content TEXT,
  reasoning TEXT,
  finish_reason TEXT,
  stream_complete INTEGER,
  ttft_ms REAL, ttfc_ms REAL, total_ms REAL,
  prompt_tokens INTEGER, completion_tokens INTEGER, reasoning_tokens INTEGER,
  tokens_estimated INTEGER, chunks INTEGER, tokens_per_s REAL,
  rating INTEGER
);
"""

RESULT_COLS = [
    "model_id",
    "rep",
    "status",
    "error",
    "http_status",
    "request_id",
    "content",
    "reasoning",
    "finish_reason",
    "stream_complete",
    "ttft_ms",
    "ttfc_ms",
    "total_ms",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "tokens_estimated",
    "chunks",
    "tokens_per_s",
]


class Connection(sqlite3.Connection):
    """sqlite3.Connection whose `with` block closes it (stdlib's only commits)."""

    def __exit__(self, *exc):
        super().__exit__(*exc)
        self.close()


def connect(path: Path | str = DB_PATH) -> Connection:
    con = sqlite3.connect(path, factory=Connection)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def create_run(con, prompt: str, system: str | None, params: dict) -> int:
    cur = con.execute(
        "INSERT INTO runs (created_at, prompt, system, params) VALUES (?, ?, ?, ?)",
        (datetime.now(UTC).isoformat(timespec="seconds"), prompt, system, json.dumps(params)),
    )
    con.commit()
    return cur.lastrowid


def save_result(con, run_id: int, r: dict) -> int:
    cols = ", ".join(RESULT_COLS)
    marks = ", ".join("?" for _ in RESULT_COLS)
    cur = con.execute(
        f"INSERT INTO results (run_id, {cols}) VALUES (?, {marks})",
        (run_id, *[r.get(c) for c in RESULT_COLS]),
    )
    con.commit()
    return cur.lastrowid


def list_runs(con, limit: int = 50) -> list[dict]:
    rows = con.execute(
        """SELECT r.id, r.created_at, r.prompt, r.params,
                  COUNT(x.id) AS n, SUM(x.status = 'error') AS errors
           FROM runs r LEFT JOIN results x ON x.run_id = r.id
           GROUP BY r.id ORDER BY r.id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(x) | {"params": json.loads(x["params"])} for x in rows]


def get_run(con, run_id: int) -> dict | None:
    run = con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if not run:
        return None
    results = con.execute("SELECT * FROM results WHERE run_id = ? ORDER BY model_id, rep", (run_id,)).fetchall()
    return dict(run) | {"params": json.loads(run["params"]), "results": [dict(x) for x in results]}


def set_rating(con, result_id: int, rating: int | None) -> None:
    con.execute("UPDATE results SET rating = ? WHERE id = ?", (rating, result_id))
    con.commit()


def delete_run(con, run_id: int) -> None:
    con.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    con.commit()
