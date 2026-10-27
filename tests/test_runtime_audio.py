import unittest
from unittest import mock

import numpy as np

from audio.beatgrid import BeatGrid
from core.automix import TransitionPhase
from core.session import DJSession
from ui.gradio_app import AirDJRuntime

SAMPLE_RATE = 44_100
BEAT_SAMPLES = 22_050


class FakePlayer:
    """Stands in for AudioPlayer without opening a sound device."""

    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails
        self.started = 0
        self._running = False
        self.audio_sinks: list = []

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self.started += 1
        if self.fails:
            raise RuntimeError("no output device")
        self._running = True

    def stop(self) -> None:
        self._running = False


def build_runtime(player: FakePlayer) -> AirDJRuntime:
    runtime = AirDJRuntime()
    runtime.player = player
    for name in ("A", "B"):
        runtime.mixer.load_deck_array(
            name, np.full((SAMPLE_RATE * 30, 2), 0.2, dtype=np.float32), SAMPLE_RATE
        )
        deck = runtime.mixer.deck_a if name == "A" else runtime.mixer.deck_b
        assert deck is not None
        deck.beat_grid = BeatGrid(
            first_beat=0.0,
            interval=float(BEAT_SAMPLES),
            total_samples=deck.frame_count,
            downbeat_offset=0,
            sample_rate=SAMPLE_RATE,
        )
        deck.bpm = deck.beat_grid.bpm
    return runtime


class RuntimeAudioTests(unittest.TestCase):
    """The transition engine is advanced from the audio callback.

    With no stream open it never advances at all: a pinch arms a transition that
    waits forever for a downbeat, the crossfader never moves, and nothing on
    screen says why. These pin the guards against that.
    """

    def test_engine_does_not_advance_without_an_audio_stream(self) -> None:
        """The failure itself, so the guards below are protecting something real."""
        from audio.mixer import Mixer

        mixer = Mixer(sample_rate=SAMPLE_RATE)
        for name in ("A", "B"):
            mixer.load_deck_array(
                name, np.full((SAMPLE_RATE * 30, 2), 0.2, dtype=np.float32), SAMPLE_RATE
            )
            deck = mixer.deck_a if name == "A" else mixer.deck_b
            assert deck is not None
            deck.beat_grid = BeatGrid(
                first_beat=0.0,
                interval=float(BEAT_SAMPLES),
                total_samples=deck.frame_count,
                downbeat_offset=0,
                sample_rate=SAMPLE_RATE,
            )
            deck.bpm = deck.beat_grid.bpm
        assert mixer.engine is not None
        session = DJSession(volume_a=1.0, volume_b=1.0)

        session.automix_requested = True
        mixer.apply_session(session)  # starts deck A
        session.automix_requested = True
        mixer.apply_session(session)  # arms the transition
        for _ in range(100):  # ten seconds of video frames, no audio blocks
            mixer.apply_session(session)

        self.assertIs(mixer.engine.phase, TransitionPhase.ARMED)
        self.assertEqual(mixer.crossfader, 0.0)

    def test_loading_a_track_opens_the_audio_output(self) -> None:
        player = FakePlayer()
        runtime = build_runtime(player)

        with mock.patch.object(runtime.mixer, "load_deck"):
            message, _, _ = runtime.load_track("A", "track.wav")

        self.assertTrue(player.is_running)
        self.assertIn("Audio output started", message)

    def test_loading_a_second_track_does_not_restart_the_output(self) -> None:
        player = FakePlayer()
        runtime = build_runtime(player)

        with mock.patch.object(runtime.mixer, "load_deck"):
            runtime.load_track("A", "a.wav")
            message, _, _ = runtime.load_track("B", "b.wav")

        self.assertEqual(player.started, 1)
        self.assertNotIn("Audio output started", message)

    def test_a_failing_output_is_reported_rather_than_swallowed(self) -> None:
        player = FakePlayer(fails=True)
        runtime = build_runtime(player)

        with mock.patch.object(runtime.mixer, "load_deck"):
            message, _, _ = runtime.load_track("A", "track.wav")

        self.assertFalse(player.is_running)
        self.assertIn("Audio output failed", message)
        self.assertIn("no output device", message)

    def test_a_pinch_opens_the_audio_output_on_demand(self) -> None:
        player = FakePlayer()
        runtime = build_runtime(player)
        runtime.session.automix_requested = True

        with mock.patch.object(runtime, "_ensure_video_components"), mock.patch.object(
            runtime, "tracker"
        ) as tracker, mock.patch.object(runtime, "overlay") as overlay:
            tracker.process_rgb.return_value = mock.Mock(hands=[], annotated_frame=None)
            overlay.draw.return_value = None
            runtime.process_frame(object())

        self.assertTrue(player.is_running)

    def test_the_status_line_names_the_real_problem(self) -> None:
        runtime = build_runtime(FakePlayer(fails=True))
        runtime.session.transition_phase = "armed"
        runtime.session.transition_label = "A to B"

        line = runtime._transition_line()

        self.assertIn("Audio output is not running", line)
        self.assertNotIn("downbeat", line)

    def test_the_status_line_is_normal_once_audio_is_live(self) -> None:
        player = FakePlayer()
        player.start()
        runtime = build_runtime(player)
        runtime.session.transition_phase = "armed"
        runtime.session.transition_label = "A to B"

        line = runtime._transition_line()

        self.assertIn("downbeat", line)


if __name__ == "__main__":
    unittest.main()
