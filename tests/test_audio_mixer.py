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
        deck_a.load_array(np.ones((4, 2), dtype=np.float32), sample_rate=44_100)
        deck_b.load_array(np.full((4, 2), 0.5, dtype=np.float32), sample_rate=44_100)
        mixer = Mixer(deck_a=deck_a, deck_b=deck_b)

        mixer.apply_session(
            DJSession(volume_a=1.0, volume_b=1.0, crossfader=0.25, playing_a=True, playing_b=True)
        )
        mixed = mixer.get_chunk(4)

        np.testing.assert_allclose(mixed, np.full((4, 2), 0.875, dtype=np.float32))

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


if __name__ == "__main__":
    unittest.main()
