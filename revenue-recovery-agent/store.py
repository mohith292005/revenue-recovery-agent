"""
store.py
Lightweight SQLite persistence layer. Solves:
- Idempotency (don't double-process the same transaction_id)
- retry_count persistence across batch runs (so the max-retry guardrail
  is real over time, not just within a single run)
- Nudge cooldown enforcement
- Survives server restarts (unlike the old in-memory-only state)
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DB_PATH = Path("data") / "recovery_state.db"
DB_PATH.parent.mkdir(exist_ok=True)


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transaction_state (
                transaction_id TEXT PRIMARY KEY,
                persisted_retry_count INTEGER NOT NULL DEFAULT 0,
                last_nudge_at TEXT,
                last_processed_at TEXT,
                total_recovered_inr REAL NOT NULL DEFAULT 0,
                times_processed INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_transaction_state(transaction_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM transaction_state WHERE transaction_id = ?",
            (transaction_id,),
        ).fetchone()
        return dict(row) if row else None


def has_been_processed(transaction_id: str) -> bool:
    """Idempotency check: has this exact transaction_id ever been processed before?"""
    state = get_transaction_state(transaction_id)
    return state is not None and state["times_processed"] > 0


def record_processing(
    transaction_id: str,
    retried: bool,
    nudged: bool,
    recovered_inr: float,
) -> None:
    """Called after a transaction is processed, to persist state for next time."""
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        existing = conn.execute(
            "SELECT * FROM transaction_state WHERE transaction_id = ?",
            (transaction_id,),
        ).fetchone()

        if existing is None:
            conn.execute(
                """INSERT INTO transaction_state
                   (transaction_id, persisted_retry_count, last_nudge_at,
                    last_processed_at, total_recovered_inr, times_processed)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    transaction_id,
                    1 if retried else 0,
                    now if nudged else None,
                    now,
                    recovered_inr,
                    1,
                ),
            )
        else:
            new_retry_count = existing["persisted_retry_count"] + (1 if retried else 0)
            new_nudge_at = now if nudged else existing["last_nudge_at"]
            conn.execute(
                """UPDATE transaction_state
                   SET persisted_retry_count = ?, last_nudge_at = ?,
                       last_processed_at = ?, total_recovered_inr = total_recovered_inr + ?,
                       times_processed = times_processed + 1
                   WHERE transaction_id = ?""",
                (new_retry_count, new_nudge_at, now, recovered_inr, transaction_id),
            )
        conn.commit()


def get_persisted_retry_count(transaction_id: str) -> int:
    state = get_transaction_state(transaction_id)
    return state["persisted_retry_count"] if state else 0


def is_within_nudge_cooldown(transaction_id: str, cooldown_hours: int) -> bool:
    state = get_transaction_state(transaction_id)
    if not state or not state["last_nudge_at"]:
        return False
    last_nudge = datetime.fromisoformat(state["last_nudge_at"])
    return datetime.utcnow() - last_nudge < timedelta(hours=cooldown_hours)


init_db()