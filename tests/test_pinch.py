import unittest

from gestures.gesture_detector import GestureDetector, HandFeatures
from gestures.pinch import PinchDetector, PinchState


class Landmark:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


def hand(ratio: float, label: str = "Right", wrist_y: float = 0.5) -> HandFeatures:
    return HandFeatures(
        label=label, wrist_x=0.5, wrist_y=wrist_y, finger_count=5, pinch_ratio=ratio
    )


def landmarks(pinch_gap: float, palm_gap: float = 0.3) -> list[Landmark]:
    """21 landmarks with a chosen thumb-to-index gap and palm size."""
    points = [Landmark(0.5, 0.9) for _ in range(21)]
    points[9] = Landmark(0.5, 0.9 - palm_gap)  # middle knuckle, above the wrist
    points[4] = Landmark(0.5, 0.85)  # thumb tip
    points[8] = Landmark(0.5, 0.85 - pinch_gap)  # index tip
    return points


class PinchRatioTests(unittest.TestCase):
    def test_ratio_is_independent_of_distance_from_the_camera(self) -> None:
        detector = GestureDetector()

        near = detector.extract("Right", landmarks(pinch_gap=0.06, palm_gap=0.30))
        far = detector.extract("Right", landmarks(pinch_gap=0.02, palm_gap=0.10))

        self.assertAlmostEqual(near.pinch_ratio, far.pinch_ratio, places=6)

    def test_ratio_corrects_for_a_non_square_frame(self) -> None:
        """On the 720x1280 portrait frame a vertical normalised unit covers 1.78x
        the distance a horizontal one does. Without correcting for that, the same
        pinch would read differently depending on how the hand was rotated."""
        detector = GestureDetector()
        aspect = 720 / 1280

        vertical_palm_horizontal_pinch = [Landmark(0.5, 0.9) for _ in range(21)]
        vertical_palm_horizontal_pinch[9] = Landmark(0.5, 0.6)
        vertical_palm_horizontal_pinch[4] = Landmark(0.5, 0.85)
        vertical_palm_horizontal_pinch[8] = Landmark(0.5 + 0.1, 0.85)

        uncorrected = detector.extract(
            "Right", vertical_palm_horizontal_pinch, aspect_ratio=1.0
        )
        corrected = detector.extract(
            "Right", vertical_palm_horizontal_pinch, aspect_ratio=aspect
        )

        # The uncorrected reading lands the wrong side of the closing threshold.
        self.assertGreater(uncorrected.pinch_ratio, 0.25)
        self.assertLess(corrected.pinch_ratio, 0.25)


class PinchDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = PinchDetector()

    def test_closing_the_pinch_fires_immediately(self) -> None:
        """The trigger cannot wait for the release: the video loop samples a few
        times a second and a quick tap can happen entirely between two frames."""
        event = self.detector.update(hand(0.1), now=10.0)

        self.assertTrue(event.closed_edge)
        self.assertIs(event.state, PinchState.CLOSED)

    def test_staying_closed_does_not_fire_again(self) -> None:
        self.detector.update(hand(0.1), now=10.0)

        second = self.detector.update(hand(0.1), now=10.1)

        self.assertFalse(second.closed_edge)
        self.assertFalse(second.holding)

    def test_a_long_pinch_is_promoted_to_a_hold_exactly_once(self) -> None:
        self.detector.update(hand(0.1), now=10.0)

        promoted = self.detector.update(hand(0.1), now=10.5)
        later = self.detector.update(hand(0.1), now=10.7)

        self.assertTrue(promoted.hold_started)
        self.assertTrue(promoted.holding)
        self.assertFalse(later.hold_started)
        self.assertTrue(later.holding)

    def test_hysteresis_stops_a_borderline_ratio_from_chattering(self) -> None:
        self.detector.update(hand(0.1), now=10.0)

        # 0.3 is above the closing threshold but below the opening one.
        between = self.detector.update(hand(0.3), now=10.1)
        self.assertIs(between.state, PinchState.CLOSED)
        self.assertFalse(between.released)

        opened = self.detector.update(hand(0.5), now=10.2)
        self.assertIs(opened.state, PinchState.OPEN)
        self.assertTrue(opened.released)

    def test_a_second_tap_inside_the_debounce_window_is_swallowed(self) -> None:
        self.detector.update(hand(0.1), now=10.0)
        self.detector.update(hand(0.5), now=10.1)

        immediate = self.detector.update(hand(0.1), now=10.2)

        self.assertFalse(immediate.closed_edge)

    def test_a_tap_after_the_debounce_window_fires_again(self) -> None:
        self.detector.update(hand(0.1), now=10.0)
        self.detector.update(hand(0.5), now=10.1)

        later = self.detector.update(hand(0.1), now=11.0)

        self.assertTrue(later.closed_edge)

    def test_hold_carries_the_hand_position(self) -> None:
        self.detector.update(hand(0.1, wrist_y=0.2), now=10.0)

        held = self.detector.update(hand(0.1, wrist_y=0.2), now=10.5)

        self.assertTrue(held.holding)
        self.assertAlmostEqual(held.hold_y, 0.2)

    def test_a_hand_leaving_the_frame_releases_its_pinch(self) -> None:
        self.detector.update(hand(0.1), now=10.0)

        released = self.detector.release("Right")

        self.assertTrue(released.released)
        self.assertNotIn("Right", self.detector.tracked_labels())

    def test_hands_are_tracked_independently(self) -> None:
        left = self.detector.update(hand(0.1, label="Left"), now=10.0)
        right = self.detector.update(hand(0.9, label="Right"), now=10.0)

        self.assertTrue(left.closed_edge)
        self.assertFalse(right.closed_edge)


if __name__ == "__main__":
    unittest.main()
