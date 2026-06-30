import hashlib
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

from app_runtime import AppPaths, AppSettings, SingleInstance, load_settings, verify_model


class RuntimeTests(unittest.TestCase):
    def test_settings_are_clamped(self):
        settings = AppSettings.from_mapping({
            "detector_confidence": 99,
            "video_target_fps": 0,
            "camera_index": -5,
            "history_limit": 1,
        })
        self.assertEqual(settings.detector_confidence, 0.95)
        self.assertEqual(settings.video_target_fps, 1.0)
        self.assertEqual(settings.camera_index, 0)
        self.assertEqual(settings.history_limit, 100)

    def test_default_config_is_created(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = AppPaths.discover(Path(temporary))
            settings = load_settings(paths)
            self.assertEqual(settings, AppSettings())
            self.assertEqual(json.loads(paths.config_file.read_text())["camera_index"], 0)

    def test_model_checksum_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.onnx"
            model.write_bytes(b"known model")
            expected = hashlib.sha256(b"known model").hexdigest()
            verify_model(model, expected)
            with self.assertRaises(RuntimeError):
                verify_model(model, "0" * 64)

    def test_corrupt_config_is_quarantined(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = AppPaths.discover(Path(temporary))
            paths.config_file.write_text("{broken", encoding="utf-8")
            with self.assertLogs("app_runtime", level="WARNING"):
                self.assertEqual(load_settings(paths), AppSettings())
            self.assertTrue(paths.config_file.with_suffix(".corrupt.json").is_file())
            self.assertEqual(json.loads(paths.config_file.read_text())["history_limit"], 10_000)

    @unittest.skipUnless(os.name == "nt", "Windows named mutex")
    def test_single_instance_mutex(self):
        name = f"VietnamLPR.Test.{uuid.uuid4()}"
        with SingleInstance(name) as first:
            self.assertFalse(first.already_running)
            with SingleInstance(name) as second:
                self.assertTrue(second.already_running)


if __name__ == "__main__":
    unittest.main()
