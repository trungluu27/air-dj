from __future__ import annotations

from dataclasses import dataclass

from audio.mixer import Mixer
from config.settings import DEFAULT_SETTINGS


@dataclass
class AudioPlayer:
    mixer: Mixer
    sample_rate: int = DEFAULT_SETTINGS.sample_rate
    block_size: int = DEFAULT_SETTINGS.block_size
    channels: int = DEFAULT_SETTINGS.output_channels

    def __post_init__(self) -> None:
        self._stream = None

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "SoundDevice is required for audio playback. Install requirements.txt."
            ) from exc

        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        try:
            stream.start()
        except Exception:
            stream.close()
            self._stream = None
            raise
        self._stream = stream

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
        finally:
            self._stream.close()
            self._stream = None

    def _callback(self, outdata, frames, _time_info, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            # Avoid logging from the real-time callback; UI/logging can surface
            # device issues during stream start instead.
            pass
        outdata[:] = self.mixer.get_chunk(frames)
