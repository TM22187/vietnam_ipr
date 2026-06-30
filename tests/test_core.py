import unittest

import numpy as np

from lpr_pipeline import OnnxPlateDetector, OnnxPlateOCR
from lpr_pipeline import clean_plate_text, find_best_model, is_valid_vietnam_plate


class FakeOcrResult:
    txts = ("30A-12345",)
    scores = (0.95,)


class FakeOcrEngine:
    def __init__(self):
        self.calls = []

    def __call__(self, _image, **kwargs):
        self.calls.append(kwargs)
        return FakeOcrResult()


class PlateTextTests(unittest.TestCase):
    def test_removes_separators(self):
        self.assertEqual(clean_plate_text("30A-123.45"), "30A12345")

    def test_repairs_common_ocr_errors(self):
        self.assertEqual(clean_plate_text("3OA12S45"), "30A12545")

    def test_accepts_common_car_and_motorcycle_formats(self):
        self.assertTrue(is_valid_vietnam_plate("30A12345"))
        self.assertTrue(is_valid_vietnam_plate("29AB12345"))

    def test_rejects_short_noise(self):
        self.assertFalse(is_valid_vietnam_plate("A123"))

    def test_packaged_model_is_available(self):
        self.assertIsNotNone(find_best_model())

    def test_detector_rejects_invalid_frame(self):
        detector = OnnxPlateDetector(find_best_model())
        with self.assertRaises(ValueError):
            detector.detect(np.zeros((10, 10), dtype=np.uint8))

    def test_wide_plate_uses_fast_recognition_path(self):
        ocr = OnnxPlateOCR.__new__(OnnxPlateOCR)
        ocr.engine = FakeOcrEngine()
        text, _, _, valid = ocr.read(np.zeros((60, 180, 3), dtype=np.uint8))
        self.assertEqual(text, "30A12345")
        self.assertTrue(valid)
        self.assertFalse(ocr.engine.calls[-1]["use_det"])

    def test_two_line_plate_keeps_text_detection(self):
        ocr = OnnxPlateOCR.__new__(OnnxPlateOCR)
        ocr.engine = FakeOcrEngine()
        ocr.read(np.zeros((120, 140, 3), dtype=np.uint8))
        self.assertTrue(ocr.engine.calls[-1]["use_det"])


if __name__ == "__main__":
    unittest.main()
