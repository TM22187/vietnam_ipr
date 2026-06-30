import tempfile
import unittest
from pathlib import Path

from history_store import HistoryStore, RecognitionRecord


def record(index: int) -> RecognitionRecord:
    return RecognitionRecord.create(
        plate=f"30A{index:05d}",
        ocr_confidence=0.91,
        detection_confidence=0.82,
        is_valid=True,
        source_type="image",
        source_name="sample.jpg",
    )


class HistoryStoreTests(unittest.TestCase):
    def test_persists_exports_and_clears(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with HistoryStore(root / "history.db") as store:
                store.add(record(1))
                store.add(record(2))
                self.assertEqual([item.plate for item in store.recent()], ["30A00002", "30A00001"])
                exported = store.export_csv(root / "history.csv")
                self.assertEqual(exported, 2)
                self.assertIn("captured_at_utc", (root / "history.csv").read_text(encoding="utf-8-sig"))
                store.clear()
                self.assertEqual(store.recent(), [])

    def test_retention_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            with HistoryStore(Path(temporary) / "history.db", retention=100) as store:
                for index in range(105):
                    store.add(record(index))
                self.assertEqual(len(store.recent(1000)), 100)

    def test_corrupt_database_is_quarantined(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "history.db"
            database.write_bytes(b"not a sqlite database")
            with self.assertLogs("history_store", level="ERROR"):
                with HistoryStore(database) as store:
                    store.add(record(1))
                    self.assertEqual(store.recent()[0].plate, "30A00001")
            self.assertEqual(len(list(Path(temporary).glob("history.corrupt-*.db"))), 1)


if __name__ == "__main__":
    unittest.main()
