"""Kho lịch sử nhận dạng bền vững bằng SQLite."""

from __future__ import annotations

import csv
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecognitionRecord:
    plate: str
    ocr_confidence: float
    detection_confidence: float
    is_valid: bool
    source_type: str
    source_name: str
    captured_at: str

    @classmethod
    def create(
        cls,
        plate: str,
        ocr_confidence: float,
        detection_confidence: float,
        is_valid: bool,
        source_type: str,
        source_name: str,
    ) -> "RecognitionRecord":
        return cls(
            plate=plate,
            ocr_confidence=max(0.0, min(1.0, float(ocr_confidence))),
            detection_confidence=max(0.0, min(1.0, float(detection_confidence))),
            is_valid=bool(is_valid),
            source_type=source_type[:32],
            source_name=source_name[:260],
            captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )


class HistoryStore:
    def __init__(self, database: str | Path, retention: int = 10_000):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.retention = max(100, retention)
        self._lock = threading.RLock()
        try:
            self._connection = self._connect()
        except sqlite3.DatabaseError:
            logger.exception("Database lịch sử bị hỏng; đang tạo database mới")
            corrupted = self.database.with_suffix(f".corrupt-{int(time.time())}.db")
            if self.database.exists():
                self.database.replace(corrupted)
            self._connection = self._connect()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=5.0, check_same_thread=False)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recognitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate TEXT NOT NULL,
                    ocr_confidence REAL NOT NULL,
                    detection_confidence REAL NOT NULL,
                    is_valid INTEGER NOT NULL,
                    source_type TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    captured_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_recognitions_captured_at "
                "ON recognitions(captured_at DESC)"
            )
            connection.commit()
            return connection
        except Exception:
            connection.close()
            raise

    def add(self, record: RecognitionRecord) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO recognitions (
                    plate, ocr_confidence, detection_confidence, is_valid,
                    source_type, source_name, captured_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.plate, record.ocr_confidence, record.detection_confidence,
                    int(record.is_valid), record.source_type, record.source_name,
                    record.captured_at,
                ),
            )
            self._connection.execute(
                "DELETE FROM recognitions WHERE id NOT IN "
                "(SELECT id FROM recognitions ORDER BY id DESC LIMIT ?)",
                (self.retention,),
            )

    def recent(self, limit: int = 50) -> list[RecognitionRecord]:
        safe_limit = max(1, min(1000, int(limit)))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT plate, ocr_confidence, detection_confidence, is_valid,
                       source_type, source_name, captured_at
                FROM recognitions ORDER BY id DESC LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [RecognitionRecord(*row[:3], bool(row[3]), *row[4:]) for row in rows]

    def clear(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM recognitions")

    def export_csv(self, destination: str | Path) -> int:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT plate, ocr_confidence, detection_confidence, is_valid,
                       source_type, source_name, captured_at
                FROM recognitions ORDER BY id DESC
                """
            ).fetchall()
        with Path(destination).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow((
                "plate", "ocr_confidence", "detection_confidence", "is_valid",
                "source_type", "source_name", "captured_at_utc",
            ))
            writer.writerows(rows)
        return len(rows)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "HistoryStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
