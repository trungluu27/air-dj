import unittest

import numpy as np

from audio.filters import SweepFilter, bass_cut_filter

SAMPLE_RATE = 44_100
BLOCK = 512


def tone(frequency: float, seconds: float = 1.0) -> np.ndarray:
    times = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    wave = np.sin(2.0 * np.pi * frequency * times).astype(np.float32)
    return np.repeat(wave[:, None], 2, axis=1)


def run_blocks(filter_: SweepFilter, signal: np.ndarray) -> np.ndarray:
    """Feed a signal through in blocks, the way the audio callback does."""
    blocks = [
        filter_.process(signal[start : start + BLOCK])
        for start in range(0, len(signal) - BLOCK, BLOCK)
    ]
    return np.concatenate(blocks)


def level_db(signal: np.ndarray) -> float:
    # Skip the filter's settling time before measuring.
    settled = signal[SAMPLE_RATE // 4 :]
    rms = float(np.sqrt(np.mean(np.square(settled, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-12) / np.sqrt(0.5))


class SweepFilterTests(unittest.TestCase):
    def test_a_centred_knob_passes_the_signal_through_untouched(self) -> None:
        sweep = SweepFilter(SAMPLE_RATE)
        block = (np.random.RandomState(0).randn(BLOCK, 2) * 0.1).astype(np.float32)

        np.testing.assert_array_equal(sweep.process(block), block)

    def test_a_positive_knob_removes_the_low_end(self) -> None:
        sweep = SweepFilter(SAMPLE_RATE)
        sweep.set_knob(1.0)

        self.assertLess(level_db(run_blocks(sweep, tone(60.0))), -30.0)

    def test_a_positive_knob_keeps_the_top_end(self) -> None:
        sweep = SweepFilter(SAMPLE_RATE)
        sweep.set_knob(1.0)

        self.assertGreater(level_db(run_blocks(sweep, tone(10_000.0))), -3.0)

    def test_a_negative_knob_removes_the_top_end(self) -> None:
        sweep = SweepFilter(SAMPLE_RATE)
        sweep.set_knob(-1.0)

        self.assertLess(level_db(run_blocks(sweep, tone(8_000.0))), -30.0)

    def test_the_sweep_moves_in_octaves(self) -> None:
        """A cutoff that moved linearly in hertz would spend most of its travel in
        a range the ear barely registers."""
        sweep = SweepFilter(SAMPLE_RATE)
        sweep.set_knob(0.5)
        midpoint = sweep.highpass_hz

        geometric_mean = float(np.sqrt(sweep.min_hp_hz * sweep.max_hp_hz))
        self.assertAlmostEqual(midpoint, geometric_mean, delta=1.0)

    def test_blocks_join_without_a_discontinuity(self) -> None:
        """State has to carry across blocks, or every 512 samples would click."""
        sweep = SweepFilter(SAMPLE_RATE)
        sweep.set_knob(0.6)
        output = run_blocks(sweep, tone(500.0))

        steps = np.abs(np.diff(output[:, 0]))
        largest_expected = float(np.median(steps)) * 12.0
        self.assertLess(float(np.max(steps[SAMPLE_RATE // 4 :])), largest_expected)

    def test_a_target_is_reached_gradually_rather_than_in_one_jump(self) -> None:
        """The gesture loop updates the knob a few times a second; stepping
        straight to each new value would be audible as zipper noise."""
        sweep = SweepFilter(SAMPLE_RATE)
        sweep.set_target(1.0)

        sweep.process(np.zeros((BLOCK, 2), dtype=np.float32))
        after_one_block = sweep.knob
        for _ in range(40):
            sweep.process(np.zeros((BLOCK, 2), dtype=np.float32))

        self.assertGreater(after_one_block, 0.0)
        self.assertLess(after_one_block, 1.0)
        self.assertAlmostEqual(sweep.knob, 1.0, places=3)

    def test_setting_the_knob_directly_does_not_glide(self) -> None:
        sweep = SweepFilter(SAMPLE_RATE)
        sweep.set_knob(0.8)

        sweep.process(np.zeros((BLOCK, 2), dtype=np.float32))

        self.assertAlmostEqual(sweep.knob, 0.8)

    def test_the_knob_is_clamped_to_its_range(self) -> None:
        sweep = SweepFilter(SAMPLE_RATE)

        sweep.set_knob(4.0)
        self.assertEqual(sweep.knob, 1.0)

        sweep.set_knob(-4.0)
        self.assertEqual(sweep.knob, -1.0)


class BassCutTests(unittest.TestCase):
    def test_the_bass_cut_removes_a_kick_without_thinning_the_track(self) -> None:
        cut = bass_cut_filter(SAMPLE_RATE)
        cut.set_knob(1.0)
        low = level_db(run_blocks(cut, tone(50.0)))

        cut = bass_cut_filter(SAMPLE_RATE)
        cut.set_knob(1.0)
        mid = level_db(run_blocks(cut, tone(2_000.0)))

        self.assertLess(low, -40.0)
        self.assertGreater(mid, -1.0)

    def test_a_zero_knob_leaves_the_bass_alone(self) -> None:
        cut = bass_cut_filter(SAMPLE_RATE)
        cut.set_knob(0.0)

        self.assertGreater(level_db(run_blocks(cut, tone(50.0))), -1.0)


if __name__ == "__main__":
    unittest.main()
