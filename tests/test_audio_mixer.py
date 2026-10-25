import unittest

import numpy as np

from audio.deck import Deck
from audio.mixer import Mixer
from core.session import DJSession


class AudioMixerTests(unittest.TestCase):
    def test_deck_returns_silence_when_paused(self) -> None:
        deck = Deck("A", sample_rate=44_100)
        deck.load_array(np.ones((4, 2), dtype=np.float32), sample_rate=44_100)

        chunk = deck.get_chunk(4)

        np.testing.assert_array_equal(chunk, np.zeros((4, 2), dtype=np.float32))

    def test_mixer_applies_deck_volumes_and_crossfader(self) -> None:
        deck_a = Deck("A", sample_rate=44_100)
        deck_b = Deck("B", sample_rate=44_100)
        deck_a.load_array(np.full((4, 2), 0.5, dtype=np.float32), sample_rate=44_100)
        deck_b.load_array(np.full((4, 2), 0.25, dtype=np.float32), sample_rate=44_100)
        mixer = Mixer(deck_a=deck_a, deck_b=deck_b)

        mixer.apply_session(
            DJSession(volume_a=1.0, volume_b=1.0, crossfader=0.25, playing_a=True, playing_b=True)
        )
        mixed = mixer.get_chunk(4)

        expected = np.cos(0.25 * np.pi / 2.0) * 0.5 + np.sin(0.25 * np.pi / 2.0) * 0.25
        np.testing.assert_allclose(mixed, np.full((4, 2), expected, dtype=np.float32))

    def test_deck_loops_when_chunk_reaches_end(self) -> None:
        deck = Deck("A", sample_rate=44_100)
        deck.load_array(
            np.array([[1.0, 1.0], [0.5, 0.5]], dtype=np.float32),
            sample_rate=44_100,
        )
        deck.play()

        chunk = deck.get_chunk(4)

        np.testing.assert_allclose(
            chunk,
            np.array(
                [[1.0, 1.0], [0.5, 0.5], [1.0, 1.0], [0.5, 0.5]],
                dtype=np.float32,
            ),
        )

    def test_mixer_loads_deck_audio_through_public_api(self) -> None:
        mixer = Mixer(sample_rate=44_100)
        audio = np.ones((4, 2), dtype=np.float32)

        mixer.load_deck_array("A", audio, sample_rate=44_100)

        self.assertTrue(mixer.deck_a.has_track)

    def test_deck_builds_waveform_summary_for_display(self) -> None:
        deck = Deck("A", sample_rate=44_100)
        audio = np.linspace(-1.0, 1.0, 44_100, dtype=np.float32)[:, None]

        deck.load_array(audio, sample_rate=44_100)

        self.assertGreater(len(deck.waveform_peaks), 0)
        self.assertLessEqual(float(np.max(deck.waveform_peaks)), 1.0)
        self.assertEqual(deck.progress, 0.0)

    def test_deck_estimates_bpm_from_regular_pulses(self) -> None:
        deck = Deck("A", sample_rate=44_100)
        audio = np.zeros((44_100 * 4, 2), dtype=np.float32)
        audio[::22_050] = 1.0

        deck.load_array(audio, sample_rate=44_100)

        self.assertAlmostEqual(deck.bpm, 120.0, delta=2.0)
        self.assertGreaterEqual(len(deck.beat_positions), 2)
        self.assertAlmostEqual(deck.beat_positions[1], 0.125, delta=0.01)

    def test_deck_finds_the_downbeat_despite_offbeat_hi_hats(self) -> None:
        """The bar line is decided by the kick, not by the busiest beat.

        Scoring downbeats on the full spectrum let hi-hats - which sit on the
        offbeats in most dance music - outvote the kick, putting the detected bar
        line a whole beat late. That is invisible on a single deck and makes two
        tracks mix a beat apart.
        """
        sample_rate = 44_100
        beat = sample_rate // 2  # 120 bpm
        audio = np.zeros((beat * 16, 2), dtype=np.float32)
        times = np.arange(beat) / sample_rate
        decay = np.exp(-np.linspace(0.0, 9.0, beat))
        kick = np.sin(2.0 * np.pi * 55.0 * times) * decay
        noise = np.random.RandomState(0).randn(beat // 8)
        hat = noise * np.exp(-np.linspace(0.0, 12.0, beat // 8)) * 0.25

        for index in range(16):
            start = index * beat
            accent = 1.6 if index % 4 == 0 else 1.0
            audio[start : start + beat] += (kick * accent)[:, None]
            if index % 2 == 1:
                audio[start : start + len(hat)] += hat[:, None]

        deck = Deck("A", sample_rate=sample_rate)
        deck.load_array(np.clip(audio, -1.0, 1.0), sample_rate=sample_rate)
        grid = deck.beat_grid

        self.assertTrue(grid.is_valid)
        declared = grid.first_beat + grid.downbeat_offset * grid.interval
        bar = grid.interval * grid.beats_per_bar
        distance = declared % bar
        self.assertLess(
            min(distance, bar - distance),
            grid.interval * 0.5,
            f"declared bar line at sample {declared:.0f} is not on a real downbeat",
        )

    def test_deck_nudge_offsets_playback_position(self) -> None:
        deck = Deck("A", sample_rate=44_100)
        deck.load_array(np.ones((10, 2), dtype=np.float32), sample_rate=44_100)

        deck.nudge(3)
        self.assertEqual(deck.position, 3.0)

        deck.nudge(-5)
        self.assertEqual(deck.position, 8.0)

    def test_mixer_applies_independent_deck_tempos_and_consumes_nudges(self) -> None:
        deck_a = Deck("A", sample_rate=44_100)
        deck_b = Deck("B", sample_rate=44_100)
        deck_a.load_array(np.ones((10, 2), dtype=np.float32), sample_rate=44_100)
        deck_b.load_array(np.ones((10, 2), dtype=np.float32), sample_rate=44_100)
        mixer = Mixer(deck_a=deck_a, deck_b=deck_b)
        session = DJSession(tempo_ratio_a=1.08, tempo_ratio_b=0.96, nudge_a=2.0, nudge_b=-1.0)

        mixer.apply_session(session)

        self.assertEqual(deck_a.tempo_ratio, 1.08)
        self.assertEqual(deck_b.tempo_ratio, 0.96)
        self.assertEqual(deck_a.position, 2.0)
        self.assertEqual(deck_b.position, 9.0)
        self.assertEqual(session.nudge_a, 0.0)
        self.assertEqual(session.nudge_b, 0.0)

    def test_mixer_syncs_slave_tempo_to_master_bpm(self) -> None:
        deck_a = Deck("A", sample_rate=44_100)
        deck_b = Deck("B", sample_rate=44_100)
        deck_a.load_array(np.ones((10, 2), dtype=np.float32), sample_rate=44_100)
        deck_b.load_array(np.ones((10, 2), dtype=np.float32), sample_rate=44_100)
        deck_a.bpm = 128.0
        deck_b.bpm = 120.0
        mixer = Mixer(deck_a=deck_a, deck_b=deck_b)
        session = DJSession(crossfader=0.25, sync_tempo_requested=True)

        mixer.apply_session(session)

        self.assertEqual(session.tempo_ratio_a, 1.0)
        self.assertAlmostEqual(session.tempo_ratio_b, 128.0 / 120.0)
        self.assertFalse(session.sync_tempo_requested)

    def test_mixer_aligns_slave_beat_phase_to_master(self) -> None:
        deck_a = Deck("A", sample_rate=44_100)
        deck_b = Deck("B", sample_rate=44_100)
        deck_a.load_array(np.ones((1_000, 2), dtype=np.float32), sample_rate=44_100)
        deck_b.load_array(np.ones((1_000, 2), dtype=np.float32), sample_rate=44_100)
        deck_a.beat_positions = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32)
        deck_b.beat_positions = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32)
        deck_a.position = 300.0
        deck_b.position = 275.0
        mixer = Mixer(deck_a=deck_a, deck_b=deck_b)
        session = DJSession(crossfader=0.25, beat_match_requested=True)

        mixer.apply_session(session)

        self.assertAlmostEqual(deck_b.position, 300.0)
        self.assertFalse(session.beat_match_requested)

    def test_deck_tempo_ratio_advances_playback_faster(self) -> None:
        deck = Deck("A", sample_rate=44_100)
        deck.load_array(
            np.array([[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75]], dtype=np.float32),
            sample_rate=44_100,
        )
        deck.tempo_ratio = 2.0
        deck.play()

        chunk = deck.get_chunk(2)

        np.testing.assert_allclose(
            chunk,
            np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float32),
        )
        self.assertEqual(deck.position, 0.0)

    def test_mixer_uses_constant_power_crossfader_curve(self) -> None:
        deck_a = Deck("A", sample_rate=44_100)
        deck_b = Deck("B", sample_rate=44_100)
        deck_a.load_array(np.full((4, 2), 0.25, dtype=np.float32), sample_rate=44_100)
        deck_b.load_array(np.full((4, 2), 0.25, dtype=np.float32), sample_rate=44_100)
        mixer = Mixer(deck_a=deck_a, deck_b=deck_b)

        mixer.apply_session(
            DJSession(volume_a=1.0, volume_b=1.0, crossfader=0.5, playing_a=True, playing_b=True)
        )
        mixed = mixer.get_chunk(4)

        np.testing.assert_allclose(
            mixed,
            np.full((4, 2), np.sqrt(0.5) * 0.5, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
