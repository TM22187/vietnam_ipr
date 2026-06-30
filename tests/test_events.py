import unittest

import numpy as np

from app_events import AppEvents


class EventTests(unittest.TestCase):
    def test_only_latest_frame_is_retained(self):
        events = AppEvents()
        events.publish("frame", np.zeros((2, 2, 3), dtype=np.uint8))
        latest = np.ones((2, 2, 3), dtype=np.uint8)
        events.publish("frame", latest)
        self.assertTrue(np.array_equal(events.take_frame(), latest))
        self.assertIsNone(events.take_frame())

    def test_control_events_keep_order(self):
        events = AppEvents()
        events.publish("status", "one")
        events.publish("done")
        self.assertEqual(events.drain(), [("status", "one"), ("done", None)])


if __name__ == "__main__":
    unittest.main()
