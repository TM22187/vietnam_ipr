import json
import unittest
from pathlib import Path

from app_runtime import APP_VERSION, MODEL_SHA256, sha256_file


ROOT = Path(__file__).resolve().parents[1]


class ReleaseConsistencyTests(unittest.TestCase):
    def test_model_manifest_and_binary_match(self):
        manifest = json.loads((ROOT / "models" / "model_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["sha256"], MODEL_SHA256)
        self.assertEqual(sha256_file(ROOT / "models" / manifest["model"]), MODEL_SHA256)

    def test_version_is_synchronized(self):
        self.assertIn(f'$Version = "{APP_VERSION}"', (ROOT / "build_app.ps1").read_text(encoding="utf-8"))
        self.assertIn(f'#define MyAppVersion "{APP_VERSION}"', (ROOT / "installer.iss").read_text(encoding="utf-8"))
        self.assertIn(f"StringStruct(u'ProductVersion', u'{APP_VERSION}')", (ROOT / "version_info.txt").read_text(encoding="utf-8"))

    def test_release_assets_exist(self):
        self.assertGreater((ROOT / "assets" / "app.ico").stat().st_size, 1024)
        self.assertTrue((ROOT / "THIRD_PARTY_NOTICES.md").is_file())
        self.assertIn("onnxruntime==", (ROOT / "requirements-lock.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
