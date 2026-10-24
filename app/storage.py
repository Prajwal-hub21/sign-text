from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Sequence


class Storage:
    """SQLite helper for persisting recognized letters and sentences."""

    def __init__(self, db_path: Path | str = Path("data/sign_texts.db")) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS letter_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                letter TEXT NOT NULL,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sentence_letters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_id INTEGER NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                letter TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def log_letter(self, letter: str, confidence: float | None = None) -> None:
        self.conn.execute(
            "INSERT INTO letter_events(letter, confidence) VALUES (?, ?);",
            (letter, confidence),
        )
        self.conn.commit()

    def save_sentence(self, content: str) -> int:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO sentences(content) VALUES (?);", (content,))
        sentence_id = cur.lastrowid
        data: Iterable[tuple[int, int, str]] = (
            (sentence_id, idx, ch) for idx, ch in enumerate(content)
        )
        cur.executemany(
            "INSERT INTO sentence_letters(sentence_id, position, letter) VALUES (?, ?, ?);",
            data,
        )
        self.conn.commit()
        return sentence_id

    def close(self) -> None:
        self.conn.close()


__all__ = ["Storage"]

