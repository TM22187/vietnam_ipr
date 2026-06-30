import unittest

from lpr_pipeline import clean_plate_text, find_best_model, is_valid_vietnam_plate


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


if __name__ == "__main__":
    unittest.main()
