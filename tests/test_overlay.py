import unittest

import numpy as np

from ui.overlay import _visible_beat_offsets, _waveform_viewport


class OverlayWaveformTests(unittest.TestCase):
    def test_waveform_viewport_centers_on_playhead_when_possible(self) -> None:
        start, end = _waveform_viewport(progress=0.5, visible_fraction=0.25)

        self.assertAlmostEqual(start, 0.375)
        self.assertAlmostEqual(end, 0.625)

    def test_waveform_viewport_clamps_at_track_edges(self) -> None:
        self.assertEqual(_waveform_viewport(progress=0.05, visible_fraction=0.25), (0.0, 0.25))
        self.assertEqual(_waveform_viewport(progress=0.95, visible_fraction=0.25), (0.75, 1.0))

    def test_visible_beat_offsets_are_relative_to_viewport(self) -> None:
        offsets = _visible_beat_offsets(
            np.array([0.25, 0.4, 0.5, 0.8], dtype=np.float32),
            start=0.35,
            end=0.65,
        )

        np.testing.assert_allclose(
            offsets,
            np.array([1.0 / 6.0, 0.5], dtype=np.float32),
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
