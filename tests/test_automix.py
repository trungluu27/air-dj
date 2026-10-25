import unittest

import numpy as np

from audio.beatgrid import BeatGrid
from audio.deck import Deck
from audio.mixer import Mixer
from core.automix import MAX_TEMPO_STRETCH, TransitionPhase
from core.session import DJSession

SAMPLE_RATE = 44_100
BEAT_SAMPLES = 11_025  # 240 bpm, so a 16 beat transition runs for four seconds
BLOCK = 512


def build_deck(name: str, beat_interval: float, level: float = 0.2) -> Deck:
    """A deck with a known grid, so the tests exercise the engine rather than the
    beat tracker."""
    deck = Deck(name, SAMPLE_RATE)
    deck.load_array(
        np.full((SAMPLE_RATE * 30, 2), level, dtype=np.float32), sample_rate=SAMPLE_RATE
    )
    deck.beat_grid = BeatGrid(
        first_beat=0.0,
        interval=float(beat_interval),
        total_samples=deck.frame_count,
        downbeat_offset=0,
        sample_rate=SAMPLE_RATE,
    )
    deck.bpm = deck.beat_grid.bpm
    return deck


class AutomixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.deck_a = build_deck("A", BEAT_SAMPLES, level=0.2)
        self.deck_b = build_deck("B", BEAT_SAMPLES, level=0.2)
        self.mixer = Mixer(sample_rate=SAMPLE_RATE, deck_a=self.deck_a, deck_b=self.deck_b)
        self.session = DJSession(volume_a=1.0, volume_b=1.0, automix_beats=16)
        assert self.mixer.engine is not None
        self.engine = self.mixer.engine

    def pinch(self) -> None:
        self.session.automix_requested = True
        self.mixer.apply_session(self.session)

    def pump(self, blocks: int) -> None:
        for _ in range(blocks):
            self.mixer.get_chunk(BLOCK)

    def run_until_idle(self, limit: int = 2_000) -> int:
        """Pump audio until the transition finishes, returning how many blocks it
        spent actually running - the arming countdown is deliberately excluded."""
        running_blocks = 0
        for _ in range(limit):
            self.mixer.get_chunk(BLOCK)
            if self.engine.phase is TransitionPhase.RUNNING:
                running_blocks += 1
            elif running_blocks:
                return running_blocks
        self.fail("transition did not finish")

    def test_first_pinch_starts_a_deck_instead_of_transitioning(self) -> None:
        self.pinch()

        self.assertTrue(self.deck_a.playing)
        self.assertFalse(self.deck_b.playing)
        self.assertIs(self.engine.phase, TransitionPhase.IDLE)
        self.assertEqual(self.mixer.crossfader, 0.0)

    def test_transition_waits_for_a_downbeat_before_anything_is_audible(self) -> None:
        self.pinch()
        self.pump(4)
        self.deck_a.position = 3.0 * BEAT_SAMPLES + 5_000  # deliberately mid-bar

        self.pinch()

        self.assertIs(self.engine.phase, TransitionPhase.ARMED)
        self.assertFalse(self.deck_b.playing)

    def test_transition_begins_on_a_bar_line(self) -> None:
        self.pinch()
        self.pump(4)
        self.deck_a.position = 3.0 * BEAT_SAMPLES + 5_000
        self.pinch()

        for _ in range(2_000):
            self.mixer.get_chunk(BLOCK)
            if self.engine.phase is TransitionPhase.RUNNING:
                break

        beat = self.deck_a.beat_grid.beat_at(self.deck_a.position)
        beats_past_the_bar = beat % self.deck_a.beat_grid.beats_per_bar
        # One audio block is the finest the timeline can resolve.
        tolerance = BLOCK / BEAT_SAMPLES
        self.assertLess(min(beats_past_the_bar, 4.0 - beats_past_the_bar), tolerance)

    def test_incoming_deck_is_cued_at_a_phrase_start(self) -> None:
        self.pinch()
        self.pump(4)
        self.deck_b.position = 7.5 * BEAT_SAMPLES
        self.pinch()
        for _ in range(2_000):
            self.mixer.get_chunk(BLOCK)
            if self.engine.phase is TransitionPhase.RUNNING:
                break

        beat = self.deck_b.beat_grid.beat_at(self.deck_b.position)
        # The deck has consumed a block of audio since it was cued, so allow for
        # that on top of the timeline's own one-block resolution.
        self.assertAlmostEqual(beat % 16.0, 0.0, delta=2.0 * BLOCK / BEAT_SAMPLES)

    def test_transition_runs_for_the_requested_number_of_beats(self) -> None:
        self.pinch()
        self.pump(4)
        self.pinch()
        blocks = self.run_until_idle()

        expected_blocks = 16 * BEAT_SAMPLES / BLOCK
        self.assertAlmostEqual(blocks, expected_blocks, delta=2)

    def test_bass_swaps_from_the_incoming_deck_at_the_halfway_beat(self) -> None:
        self.pinch()
        self.pump(4)
        self.pinch()

        before_swap: tuple[float, float] | None = None
        after_swap: tuple[float, float] | None = None
        for _ in range(2_000):
            self.mixer.get_chunk(BLOCK)
            if self.engine.phase is not TransitionPhase.RUNNING:
                continue
            if self.engine.progress < 0.3:
                before_swap = (self.engine.bass_cut_a, self.engine.bass_cut_b)
            elif 0.7 < self.engine.progress < 0.95:
                after_swap = (self.engine.bass_cut_a, self.engine.bass_cut_b)

        self.assertEqual(before_swap, (0.0, 1.0), "incoming deck B should start bass-cut")
        self.assertEqual(after_swap, (1.0, 0.0), "the cut should move to the outgoing deck")

    def test_transition_hands_over_playback_and_ends_on_the_incoming_deck(self) -> None:
        self.pinch()
        self.pump(4)
        self.pinch()
        self.run_until_idle()

        self.assertFalse(self.deck_a.playing)
        self.assertTrue(self.deck_b.playing)
        self.assertEqual(self.mixer.crossfader, 1.0)
        self.assertEqual(self.engine.bass_cut_a, 0.0)
        self.assertEqual(self.engine.bass_cut_b, 0.0)

    def test_crossfader_moves_only_forwards_across_the_transition(self) -> None:
        self.pinch()
        self.pump(4)
        self.pinch()

        positions: list[float] = []
        for _ in range(2_000):
            self.mixer.get_chunk(BLOCK)
            if self.engine.phase is TransitionPhase.RUNNING:
                positions.append(self.engine.crossfader)
            elif positions:
                break

        self.assertGreater(len(positions), 10)
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(positions, positions[1:])))
        self.assertAlmostEqual(positions[0], 0.0, delta=0.05)

    def test_incoming_tempo_is_matched_to_the_outgoing_deck(self) -> None:
        # Deck B is three percent slower, which is inside the stretch window.
        self.deck_b = build_deck("B", BEAT_SAMPLES * 1.03)
        self.mixer = Mixer(sample_rate=SAMPLE_RATE, deck_a=self.deck_a, deck_b=self.deck_b)
        assert self.mixer.engine is not None
        self.engine = self.mixer.engine

        self.pinch()
        self.pump(4)
        self.pinch()
        for _ in range(2_000):
            self.mixer.get_chunk(BLOCK)
            if self.engine.phase is TransitionPhase.RUNNING:
                break

        self.assertAlmostEqual(
            self.deck_b.bpm * self.deck_b.tempo_ratio, self.deck_a.bpm, delta=0.5
        )

    def test_a_wildly_different_tempo_is_capped_rather_than_pitch_shifted(self) -> None:
        self.deck_b = build_deck("B", BEAT_SAMPLES * 1.4)
        self.mixer = Mixer(sample_rate=SAMPLE_RATE, deck_a=self.deck_a, deck_b=self.deck_b)
        assert self.mixer.engine is not None
        self.engine = self.mixer.engine

        self.pinch()
        self.pump(4)
        self.pinch()
        for _ in range(2_000):
            self.mixer.get_chunk(BLOCK)
            if self.engine.phase is TransitionPhase.RUNNING:
                break

        self.assertLessEqual(self.deck_b.tempo_ratio, 1.0 + MAX_TEMPO_STRETCH + 1e-9)
        self.assertIn("capped", self.engine.status)

    def test_half_time_counts_as_matched_without_stretching(self) -> None:
        # Deck B at half the tempo sits perfectly under deck A, so the engine
        # should leave it alone rather than trying to stretch it by 100%.
        self.deck_b = build_deck("B", BEAT_SAMPLES * 2.0)
        self.mixer = Mixer(sample_rate=SAMPLE_RATE, deck_a=self.deck_a, deck_b=self.deck_b)
        assert self.mixer.engine is not None
        self.engine = self.mixer.engine

        self.pinch()
        self.pump(4)
        self.pinch()
        for _ in range(2_000):
            self.mixer.get_chunk(BLOCK)
            if self.engine.phase is TransitionPhase.RUNNING:
                break

        self.assertAlmostEqual(self.deck_b.tempo_ratio, 1.0, delta=1e-6)
        self.assertNotIn("capped", self.engine.status)

    def test_quieter_incoming_track_is_levelled_up(self) -> None:
        self.deck_b = build_deck("B", BEAT_SAMPLES, level=0.12)
        self.mixer = Mixer(sample_rate=SAMPLE_RATE, deck_a=self.deck_a, deck_b=self.deck_b)
        assert self.mixer.engine is not None
        self.engine = self.mixer.engine

        self.pinch()
        self.pump(4)
        self.pinch()
        for _ in range(2_000):
            self.mixer.get_chunk(BLOCK)
            if self.engine.phase is TransitionPhase.RUNNING:
                break

        self.assertAlmostEqual(self.deck_b.rms * self.engine.trim_b, self.deck_a.rms, delta=1e-3)

    def test_second_pinch_during_a_transition_is_ignored(self) -> None:
        self.pinch()
        self.pump(4)
        self.pinch()
        self.pump(200)
        progress = self.engine.progress
        direction = (self.engine.from_deck, self.engine.to_deck)

        self.pinch()

        self.assertEqual((self.engine.from_deck, self.engine.to_deck), direction)
        self.assertGreaterEqual(self.engine.progress, progress)

    def test_cancelling_while_armed_leaves_the_mix_untouched(self) -> None:
        self.pinch()
        self.pump(4)
        self.deck_a.position = 5_000
        self.pinch()
        self.assertIs(self.engine.phase, TransitionPhase.ARMED)

        self.session.automix_cancel_requested = True
        self.mixer.apply_session(self.session)
        self.pump(40)

        self.assertIs(self.engine.phase, TransitionPhase.IDLE)
        self.assertTrue(self.deck_a.playing)
        self.assertFalse(self.deck_b.playing)
        self.assertEqual(self.mixer.crossfader, 0.0)

    def test_transition_alternates_direction(self) -> None:
        self.pinch()
        self.pump(4)
        self.pinch()
        self.run_until_idle()
        self.assertEqual((self.engine.from_deck, self.engine.to_deck), ("A", "B"))

        self.pinch()
        self.run_until_idle()

        self.assertEqual((self.engine.from_deck, self.engine.to_deck), ("B", "A"))
        self.assertTrue(self.deck_a.playing)
        self.assertFalse(self.deck_b.playing)

    def test_request_is_refused_when_the_other_deck_is_empty(self) -> None:
        empty = Deck("B", SAMPLE_RATE)
        mixer = Mixer(sample_rate=SAMPLE_RATE, deck_a=self.deck_a, deck_b=empty)
        assert mixer.engine is not None
        session = DJSession(automix_requested=True)
        mixer.apply_session(session)  # cold start on A
        session.automix_requested = True

        mixer.apply_session(session)

        self.assertIs(mixer.engine.phase, TransitionPhase.IDLE)
        self.assertIn("load a track", mixer.engine.status)

    def test_session_reports_transition_state_for_the_ui(self) -> None:
        self.pinch()
        self.pump(4)
        self.pinch()
        self.pump(200)
        self.mixer.apply_session(self.session)

        self.assertEqual(self.session.transition_phase, "running")
        self.assertGreater(self.session.transition_progress, 0.0)
        self.assertLess(self.session.transition_beats_remaining, 16.0)
        self.assertEqual(self.session.transition_label, "A to B")

    def test_output_stays_within_range_across_a_whole_transition(self) -> None:
        self.pinch()
        self.pump(4)
        self.pinch()

        peak = 0.0
        for _ in range(2_000):
            chunk = self.mixer.get_chunk(BLOCK)
            peak = max(peak, float(np.max(np.abs(chunk))))
            if self.engine.phase is TransitionPhase.IDLE and self.engine.progress >= 1.0:
                break

        self.assertLessEqual(peak, 1.0)


if __name__ == "__main__":
    unittest.main()
