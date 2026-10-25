import unittest

from core.session import DJSession
from gestures.gesture_detector import HandFeatures
from gestures.gesture_mapping import GestureMapper


def hand(
    ratio: float, label: str = "Right", wrist_y: float = 0.5, wrist_x: float = 0.5
) -> HandFeatures:
    return HandFeatures(
        label=label,
        wrist_x=wrist_x,
        wrist_y=wrist_y,
        finger_count=5,
        pinch_ratio=ratio,
    )


OPEN = 0.9
CLOSED = 0.1


class GestureMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = DJSession()
        self.mapper = GestureMapper(smoothing_alpha=1.0)

    def test_pinch_asks_for_a_transition(self) -> None:
        self.mapper.apply(self.session, [hand(CLOSED)], now=10.0)

        self.assertTrue(self.session.automix_requested)

    def test_an_open_hand_asks_for_nothing(self) -> None:
        """The old mapping played a deck whenever a palm was open, which meant
        simply raising a hand started the music."""
        self.mapper.apply(self.session, [hand(OPEN)], now=10.0)

        self.assertFalse(self.session.automix_requested)
        self.assertEqual(self.session.filter_master, 0.0)

    def test_holding_the_pinch_still_mixes_as_long_as_the_hand_stays_put(self) -> None:
        """A slow pinch has to behave like a quick one.

        Holding alone used to cancel the transition after a third of a second,
        which made an ordinary unhurried pinch do nothing at all - no mix, and no
        filter either, because a hand resting mid-frame is inside the deadzone.
        """
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.5)], now=10.0)
        self.assertTrue(self.session.automix_requested)

        for moment in (10.4, 10.8, 11.5, 12.0):
            self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.5)], now=moment)

        self.assertTrue(self.session.automix_requested)
        self.assertFalse(self.session.automix_cancel_requested)

    def test_moving_the_hand_while_pinched_withdraws_the_transition(self) -> None:
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.5)], now=10.0)
        self.session.automix_requested = True

        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.1)], now=10.6)

        self.assertFalse(self.session.automix_requested)
        self.assertTrue(self.session.automix_cancel_requested)

    def test_pinch_state_is_published_for_the_overlay(self) -> None:
        """The meter on screen is the only way to tell a pinch that just missed the
        threshold from no gesture at all."""
        self.mapper.apply(self.session, [], now=10.0)
        self.assertEqual(self.session.pinch_state, "no hand")

        self.mapper.apply(self.session, [hand(OPEN)], now=10.1)
        self.assertEqual(self.session.pinch_state, "open")

        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.5)], now=10.2)
        self.assertEqual(self.session.pinch_state, "pinched")

        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.5)], now=10.8)
        self.assertEqual(self.session.pinch_state, "hold")

        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.1)], now=11.0)
        self.assertEqual(self.session.pinch_state, "sweeping")

    def test_holding_high_opens_the_high_pass(self) -> None:
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.5)], now=10.0)

        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.0)], now=10.6)

        self.assertAlmostEqual(self.session.filter_master, 1.0)

    def test_holding_low_closes_the_low_pass(self) -> None:
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.5)], now=10.0)

        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=1.0)], now=10.6)

        self.assertAlmostEqual(self.session.filter_master, -1.0)

    def test_the_middle_of_the_frame_is_a_deadzone(self) -> None:
        """A hold meant only to cancel a transition should not also sweep the
        filter, so the hand has to travel out of the centre first."""
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.5)], now=10.0)

        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.53)], now=10.6)

        self.assertEqual(self.session.filter_master, 0.0)

    def test_releasing_the_pinch_returns_the_filter_to_centre(self) -> None:
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.5)], now=10.0)
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.0)], now=10.6)
        self.assertAlmostEqual(self.session.filter_master, 1.0)

        self.mapper.apply(self.session, [hand(OPEN, wrist_y=0.0)], now=10.8)

        self.assertEqual(self.session.filter_master, 0.0)

    def test_a_hand_leaving_the_frame_returns_the_filter_to_centre(self) -> None:
        """Nothing may stay latched on when tracking is lost - being unable to let
        go of a control is what made the old mapping tiring to use."""
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.5)], now=10.0)
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.0)], now=10.6)

        self.mapper.apply(self.session, [], now=10.8)

        self.assertEqual(self.session.filter_master, 0.0)

    def test_one_hand_alone_is_enough(self) -> None:
        """The old crossfader needed both wrists in frame at once."""
        self.mapper.apply(self.session, [hand(CLOSED, label="Left")], now=10.0)

        self.assertTrue(self.session.automix_requested)

    def test_the_right_hand_leads_when_both_are_up(self) -> None:
        self.mapper.apply(
            self.session,
            [hand(OPEN, label="Right"), hand(CLOSED, label="Left")],
            now=10.0,
        )

        self.assertFalse(self.session.automix_requested)

    def test_hand_height_no_longer_moves_the_volumes(self) -> None:
        self.mapper.apply(
            self.session,
            [hand(OPEN, label="Left", wrist_y=0.0), hand(OPEN, label="Right", wrist_y=1.0)],
            now=10.0,
        )

        self.assertEqual(self.session.volume_a, 0.75)
        self.assertEqual(self.session.volume_b, 0.75)

    def test_two_hands_no_longer_move_the_crossfader(self) -> None:
        self.mapper.apply(
            self.session,
            [
                hand(OPEN, label="Left", wrist_x=0.1),
                hand(OPEN, label="Right", wrist_x=0.3),
            ],
            now=10.0,
        )

        self.assertEqual(self.session.crossfader, 0.5)

    def test_filter_is_smoothed_between_frames(self) -> None:
        mapper = GestureMapper(smoothing_alpha=0.5)
        mapper.apply(self.session, [hand(CLOSED, wrist_y=0.5)], now=10.0)

        mapper.apply(self.session, [hand(CLOSED, wrist_y=0.0)], now=10.6)

        self.assertAlmostEqual(self.session.filter_master, 0.5)


if __name__ == "__main__":
    unittest.main()
