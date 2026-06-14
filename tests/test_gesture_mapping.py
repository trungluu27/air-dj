import time
import unittest

from core.session import DJSession
from gestures.gesture_classifier import GestureType
from gestures.gesture_detector import HandFeatures
from gestures.gesture_mapping import GestureMapper


class GestureMapperTests(unittest.TestCase):
    def test_left_and_right_wrist_height_control_deck_volumes(self) -> None:
        session = DJSession()
        mapper = GestureMapper(smoothing_alpha=1.0)

        mapper.apply(
            session,
            [
                HandFeatures(label="Left", wrist_x=0.25, wrist_y=0.0, finger_count=5),
                HandFeatures(label="Right", wrist_x=0.75, wrist_y=1.0, finger_count=5),
            ],
            now=time.monotonic(),
        )

        self.assertEqual(session.volume_a, 1.0)
        self.assertEqual(session.volume_b, 0.0)

    def test_two_hands_control_crossfader_from_midpoint(self) -> None:
        session = DJSession()
        mapper = GestureMapper(smoothing_alpha=1.0)

        mapper.apply(
            session,
            [
                HandFeatures(label="Left", wrist_x=0.2, wrist_y=0.5, finger_count=5),
                HandFeatures(label="Right", wrist_x=0.6, wrist_y=0.5, finger_count=5),
            ],
            now=time.monotonic(),
        )

        self.assertAlmostEqual(session.crossfader, 0.4)

    def test_open_palm_and_fist_toggle_assigned_deck_with_cooldown(self) -> None:
        session = DJSession()
        mapper = GestureMapper(smoothing_alpha=1.0, play_pause_cooldown_seconds=0.8)

        mapper.apply(
            session,
            [HandFeatures(label="Left", wrist_x=0.2, wrist_y=0.5, finger_count=5)],
            now=10.0,
        )
        self.assertTrue(session.playing_a)
        self.assertEqual(session.last_gesture_a, GestureType.OPEN_PALM)

        mapper.apply(
            session,
            [HandFeatures(label="Left", wrist_x=0.2, wrist_y=0.5, finger_count=0)],
            now=10.1,
        )
        self.assertTrue(session.playing_a)

        mapper.apply(
            session,
            [HandFeatures(label="Left", wrist_x=0.2, wrist_y=0.5, finger_count=0)],
            now=11.0,
        )
        self.assertFalse(session.playing_a)
        self.assertEqual(session.last_gesture_a, GestureType.FIST)


if __name__ == "__main__":
    unittest.main()
