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

    def test_small_movements_sit_inside_a_deadzone(self) -> None:
        """Tremor is not a gesture."""
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.5)], now=10.0)

        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.53)], now=10.6)

        self.assertEqual(self.session.filter_master, 0.0)

    def test_holding_still_off_centre_does_not_cancel_the_mix(self) -> None:
        """The sweep is measured from where the pinch closed, not from the middle
        of the frame.

        Anchoring to the frame meant a hand held perfectly still anywhere but dead
        centre read as a deliberate sweep. A pinch at chest height - which is where
        a hand naturally sits - cancelled its own transition the moment it became a
        hold, so the crossfader never moved and nothing said why.
        """
        for height in (0.20, 0.30, 0.35, 0.42, 0.50, 0.65, 0.80):
            session = DJSession()
            mapper = GestureMapper(smoothing_alpha=1.0)
            still = hand(CLOSED, wrist_y=height)

            mapper.apply(session, [still], now=10.0)
            for moment in (10.4, 10.8, 11.2):
                mapper.apply(session, [still], now=moment)

            self.assertTrue(
                session.automix_requested, f"a still hand at y={height} cancelled its mix"
            )
            self.assertFalse(session.automix_cancel_requested)
            self.assertEqual(session.filter_master, 0.0)
            self.assertEqual(session.pinch_state, "hold")

    def test_the_sweep_is_relative_to_wherever_the_pinch_started(self) -> None:
        for anchor in (0.25, 0.5, 0.75):
            session = DJSession()
            mapper = GestureMapper(smoothing_alpha=1.0)
            mapper.apply(session, [hand(CLOSED, wrist_y=anchor)], now=10.0)

            mapper.apply(session, [hand(CLOSED, wrist_y=anchor - 0.14)], now=10.6)

            self.assertGreater(session.filter_master, 0.3, f"anchored at {anchor}")
            self.assertLess(session.filter_master, 0.9, f"anchored at {anchor}")

    def test_releasing_the_pinch_returns_the_filter_to_centre(self) -> None:
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.5)], now=10.0)
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.0)], now=10.6)
        self.assertAlmostEqual(self.session.filter_master, 1.0)

        self.mapper.apply(self.session, [hand(OPEN, wrist_y=0.0)], now=10.8)

        self.assertEqual(self.session.filter_master, 0.0)

    def test_a_hand_leaving_the_frame_returns_the_filter_to_centre(self) -> None:
        """Nothing may stay latched on once the hand is really gone - being unable
        to let go of a control is what made the old mapping tiring to use.

        The release waits out the dropped-frame grace first; see
        test_a_dropped_frame_does_not_release_the_hand for the other half.
        """
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.5)], now=10.0)
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.0)], now=10.6)

        self.mapper.apply(self.session, [], now=11.4)

        self.assertEqual(self.session.filter_master, 0.0)
        self.assertEqual(self.session.active_hand, "")

    def test_one_hand_alone_is_enough(self) -> None:
        """The old crossfader needed both wrists in frame at once."""
        self.mapper.apply(self.session, [hand(CLOSED, label="Left")], now=10.0)

        self.assertTrue(self.session.automix_requested)
        self.assertEqual(self.session.active_hand, "Left")

    def test_the_right_hand_leads_when_both_arrive_together(self) -> None:
        self.mapper.apply(
            self.session,
            [hand(OPEN, label="Right"), hand(CLOSED, label="Left")],
            now=10.0,
        )

        self.assertEqual(self.session.active_hand, "Right")
        self.assertFalse(self.session.automix_requested)

    def test_a_second_hand_cannot_take_over_from_the_first(self) -> None:
        """Control stays where the user put it.

        The right hand used to win every frame, so a second hand drifting into
        view seized control mid-gesture.
        """
        self.mapper.apply(self.session, [hand(OPEN, label="Left")], now=10.0)
        self.assertEqual(self.session.active_hand, "Left")

        # The right hand appears, pinching. It must be ignored entirely.
        self.mapper.apply(
            self.session,
            [hand(OPEN, label="Left"), hand(CLOSED, label="Right")],
            now=10.2,
        )

        self.assertEqual(self.session.active_hand, "Left")
        self.assertFalse(self.session.automix_requested)

    def test_the_locked_hand_still_drives_while_a_second_is_in_frame(self) -> None:
        self.mapper.apply(self.session, [hand(OPEN, label="Left")], now=10.0)

        self.mapper.apply(
            self.session,
            [hand(CLOSED, label="Left"), hand(OPEN, label="Right")],
            now=10.2,
        )

        self.assertTrue(self.session.automix_requested)
        self.assertEqual(self.session.active_hand, "Left")

    def test_a_dropped_frame_does_not_release_the_hand(self) -> None:
        """MediaPipe loses a hand for a frame now and then; letting go of the
        filter over a blink would make a sweep stutter."""
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.5)], now=10.0)
        self.mapper.apply(self.session, [hand(CLOSED, wrist_y=0.0)], now=10.6)
        swept = self.session.filter_master
        self.assertAlmostEqual(swept, 1.0)

        self.mapper.apply(self.session, [], now=10.7)

        self.assertEqual(self.session.filter_master, swept)
        self.assertEqual(self.session.active_hand, "Right")

    def test_control_is_released_once_the_hand_is_really_gone(self) -> None:
        self.mapper.apply(self.session, [hand(CLOSED, label="Left")], now=10.0)

        self.mapper.apply(self.session, [], now=11.0)

        self.assertEqual(self.session.active_hand, "")
        self.assertEqual(self.session.pinch_state, "no hand")

    def test_the_other_hand_can_take_over_after_the_first_leaves(self) -> None:
        self.mapper.apply(self.session, [hand(OPEN, label="Left")], now=10.0)
        self.mapper.apply(self.session, [], now=11.0)

        self.mapper.apply(self.session, [hand(CLOSED, label="Right")], now=11.1)

        self.assertEqual(self.session.active_hand, "Right")
        self.assertTrue(self.session.automix_requested)

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
