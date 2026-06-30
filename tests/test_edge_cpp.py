import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EDGE = ROOT / "edge_cpp"


class EdgeCppReleaseTests(unittest.TestCase):
    def test_required_sources_exist(self):
        for relative in (
            "CMakeLists.txt",
            "src/main.cpp",
            "src/engine.cpp",
            "src/text.cpp",
            "systemd/vietnam-lpr-edge.service",
        ):
            self.assertTrue((EDGE / relative).is_file(), relative)

    def test_manifest_files_and_hashes(self):
        manifest = json.loads((EDGE / "models/model_manifest.json").read_text("utf-8"))
        locations = {
            "best_vietnam_lpr.onnx": ROOT / "models/best_vietnam_lpr.onnx",
            "PP-OCRv6_rec_small.onnx": EDGE / "models/PP-OCRv6_rec_small.onnx",
            "ppocrv6_chars.txt": EDGE / "models/ppocrv6_chars.txt",
        }
        for item in manifest["models"]:
            path = locations[item["file"]]
            self.assertEqual(path.stat().st_size, item["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])

    def test_charset_contract(self):
        characters = (EDGE / "models/ppocrv6_chars.txt").read_text("utf-8").splitlines()
        self.assertEqual(len(characters), 18708)
        self.assertEqual(len(characters) + 2, 18710)  # CTC blank + trailing space

    def test_runtime_is_headless(self):
        source = "\n".join(
            (EDGE / relative).read_text("utf-8")
            for relative in ("src/main.cpp", "src/engine.cpp")
        ).lower()
        self.assertNotIn("imshow", source)
        self.assertNotIn("selectroi", source)


if __name__ == "__main__":
    unittest.main()
