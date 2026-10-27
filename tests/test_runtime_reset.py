import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio.beatgrid import BeatGrid
from audio.player import AudioPlayer
from core.automix import TransitionPhase
from gestures.gesture_detector import HandFeatures
from recording.recorder import VideoAudioRecorder
from ui.gradio_app import AirDJRuntime

SAMPLE_RATE = 44_100
BEAT_SAMPLES = 22_050

# message, the eight values a frame publishes, transition beats, the two upload
# slots and the recording download.
RESET_OUTPUT_COUNT = 13


class FakePlayer:
    """Stands in for AudioPlayer without opening a sound device."""

    def __init__(self) -> None:
        self._running = False
        self.audio_sinks: list = []
        self.stopped = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self.stopped += 1
        self._running = False


class FakeCapture:
    def __init__(self) -> None:
        self.released = 0

    def release(self) -> None:
        self.released += 1


def pinching_hand(label: str = "Right") -> HandFeatures:
    return HandFeatures(
        label=label, wrist_x=0.5, wrist_y=0.5, finger_count=5, pinch_ratio=0.1
    )


class RuntimeResetTests(unittest.TestCase):
    """One button has to undo everything, including the state that is not visible.

    A reset that leaves a transition armed, a beat grid loaded, a filter's biquad
    charged or a hand locked in the mapper is worse than no reset at all: the
    screen looks like a fresh start and the app does not behave like one.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.runtime = AirDJRuntime()
        self.player = FakePlayer()
        self.runtime.player = self.player
        self.runtime.recorder = VideoAudioRecorder(Path(self.tmpdir.name))
        self.player.audio_sinks.append(self.runtime.recorder.write_audio)

    def load_decks(self) -> None:
        for name in ("A", "B"):
            self.runtime.mixer.load_deck_array(
                name,
                np.full((SAMPLE_RATE * 30, 2), 0.2, dtype=np.float32),
                SAMPLE_RATE,
            )
            deck = self.runtime.mixer.deck_a if name == "A" else self.runtime.mixer.deck_b
            assert deck is not None
            deck.track_name = f"track-{name}.wav"
            deck.beat_grid = BeatGrid(
                first_beat=0.0,
                interval=float(BEAT_SAMPLES),
                total_samples=deck.frame_count,
                downbeat_offset=0,
                sample_rate=SAMPLE_RATE,
            )
            deck.bpm = deck.beat_grid.bpm

    def arm_a_transition(self) -> None:
        session = self.runtime.session
        session.automix_requested = True
        self.runtime.mixer.apply_session(session)  # starts deck A
        session.automix_requested = True
        self.runtime.mixer.apply_session(session)  # arms the mix

    def test_reset_unloads_both_decks(self) -> None:
        self.load_decks()

        self.runtime.reset()

        assert self.runtime.mixer.deck_a is not None
        assert self.runtime.mixer.deck_b is not None
        self.assertFalse(self.runtime.mixer.deck_a.has_track)
        self.assertFalse(self.runtime.mixer.deck_b.has_track)

    def test_reset_abandons_a_transition_in_flight(self) -> None:
        self.load_decks()
        self.arm_a_transition()
        assert self.runtime.mixer.engine is not None
        self.assertIsNot(self.runtime.mixer.engine.phase, TransitionPhase.IDLE)

        self.runtime.reset()

        assert self.runtime.mixer.engine is not None
        self.assertIs(self.runtime.mixer.engine.phase, TransitionPhase.IDLE)
        self.assertFalse(self.runtime.mixer.engine.owns_transport)
        self.assertEqual(self.runtime.mixer.crossfader, 0.5)
        self.assertEqual(self.runtime.session.transition_phase, "idle")

    def test_reset_stops_the_audio_output(self) -> None:
        self.player.start()

        self.runtime.reset()

        self.assertEqual(self.player.stopped, 1)
        self.assertFalse(self.player.is_running)

    def test_reset_stops_the_camera(self) -> None:
        capture = FakeCapture()
        self.runtime.camera_capture = capture

        self.runtime.reset()

        self.assertEqual(capture.released, 1)
        self.assertIsNone(self.runtime.camera_capture)

    def test_reset_lets_go_of_the_locked_hand(self) -> None:
        """The mapper follows the first hand it sees until that hand leaves.

        Carried across a reset, the next hand in frame would be ignored with
        nothing to say why.
        """
        self.runtime.mapper.apply(self.runtime.session, [pinching_hand("Left")], now=10.0)
        self.assertEqual(self.runtime.session.active_hand, "Left")

        self.runtime.reset()

        self.runtime.mapper.apply(self.runtime.session, [pinching_hand("Right")], now=11.0)
        self.assertEqual(self.runtime.session.active_hand, "Right")
        self.assertTrue(self.runtime.session.automix_requested)

    def test_reset_returns_the_session_to_its_defaults(self) -> None:
        session = self.runtime.session
        session.filter_master = -0.8
        session.automix_beats = 32
        session.pinch_state = "sweeping"
        session.crossfader = 1.0

        message, *_, beats, track_a, track_b, _ = self.runtime.reset()

        self.assertEqual(self.runtime.session.filter_master, 0.0)
        self.assertEqual(self.runtime.session.automix_beats, 16)
        self.assertEqual(self.runtime.session.pinch_state, "no hand")
        self.assertEqual(self.runtime.session.crossfader, 0.5)
        self.assertEqual(beats, 16)
        self.assertIsNone(track_a)
        self.assertIsNone(track_b)
        self.assertIn("Reset", message)

    def test_reset_rebuilds_the_player_around_the_new_mixer(self) -> None:
        """Audio has to still work afterwards.

        The player holds the mixer by reference and the recorder by sink, so one
        left pointing at the discarded pair would play silence into nothing.
        """
        self.runtime.reset()

        player = self.runtime.player
        assert isinstance(player, AudioPlayer)
        self.assertIs(player.mixer, self.runtime.mixer)
        self.assertEqual(len(player.audio_sinks), 1)

        block = np.ones((512, 2), dtype=np.float32)
        self.runtime.recorder.start()
        player.audio_sinks[0](block)
        self.assertGreater(self.runtime.recorder.stop().audio_sample_count, 0)

    def test_reset_keeps_a_recording_that_was_still_running(self) -> None:
        """A half-written take is the user's work, not state to throw away."""
        self.runtime.recorder.start()
        self.runtime.recorder.write_audio(np.ones((512, 2), dtype=np.float32))

        message, *_, download = self.runtime.reset()

        self.assertFalse(self.runtime.recorder.is_recording)
        self.assertIn("recording saved", message)
        self.assertIsNotNone(download)
        self.assertEqual(self.runtime.recorder.output_dir, Path(self.tmpdir.name))

    def test_reset_clears_the_download_when_nothing_was_recording(self) -> None:
        *_, download = self.runtime.reset()

        self.assertIsNone(download)
        self.assertIsNone(self.runtime.last_recording_path)

    def test_reset_publishes_one_value_per_control_it_owns(self) -> None:
        outputs = self.runtime.reset()

        self.assertEqual(len(outputs), RESET_OUTPUT_COUNT)

    def test_reset_shows_the_starting_camera_placeholder(self) -> None:
        _, frame, status, *_ = self.runtime.reset()

        self.assertIsNotNone(frame)
        self.assertEqual(status["pinch_state"], "no hand")
        self.assertEqual(status["transition_phase"], "idle")


if __name__ == "__main__":
    unittest.main()
