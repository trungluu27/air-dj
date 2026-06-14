import sys
import types
import unittest

import numpy as np

from audio.mixer import Mixer
from audio.player import AudioPlayer
from ui.gradio_app import AirDJRuntime


class _FailingStream:
    starts = 0
    closes = 0

    def __init__(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    def start(self) -> None:
        type(self).starts += 1
        raise RuntimeError("device unavailable")

    def stop(self) -> None:
        pass

    def close(self) -> None:
        type(self).closes += 1


class _StartedPlayer:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class AudioPlayerTests(unittest.TestCase):
    def test_start_failure_clears_stream_so_retry_can_start_again(self) -> None:
        original = sys.modules.get("sounddevice")
        fake_module = types.SimpleNamespace(OutputStream=_FailingStream)
        sys.modules["sounddevice"] = fake_module  # type: ignore[assignment]
        _FailingStream.starts = 0
        _FailingStream.closes = 0

        try:
            player = AudioPlayer(Mixer())

            with self.assertRaisesRegex(RuntimeError, "device unavailable"):
                player.start()
            with self.assertRaisesRegex(RuntimeError, "device unavailable"):
                player.start()

            self.assertEqual(_FailingStream.starts, 2)
            self.assertEqual(_FailingStream.closes, 2)
        finally:
            if original is None:
                sys.modules.pop("sounddevice", None)
            else:
                sys.modules["sounddevice"] = original

    def test_start_audio_explains_silence_when_no_track_is_loaded(self) -> None:
        runtime = AirDJRuntime()
        runtime.player = _StartedPlayer()  # type: ignore[assignment]

        message = runtime.start_audio()

        self.assertIn("Audio output started", message)
        self.assertIn("No track is loaded", message)

    def test_start_audio_explains_silence_when_track_is_loaded_but_paused(self) -> None:
        runtime = AirDJRuntime()
        runtime.player = _StartedPlayer()  # type: ignore[assignment]
        runtime.mixer.load_deck_array(
            "A",
            np.ones((4, 2), dtype=np.float32),
            sample_rate=44_100,
        )

        message = runtime.start_audio()

        self.assertIn("Audio output started", message)
        self.assertIn("No deck is playing", message)
        self.assertIn("open palm", message)


if __name__ == "__main__":
    unittest.main()
