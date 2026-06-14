from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Deck:
    name: str
    sample_rate: int
    volume: float = 1.0
    playing: bool = False
    position: int = 0
    track_name: str = "No track"
    _audio: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float32),
        repr=False,
    )

    def load(self, path: str | Path) -> None:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "SoundFile is required for loading audio. Install requirements.txt."
            ) from exc

        audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
        self.load_array(audio, sample_rate=source_rate)
        self.track_name = Path(path).name

    def load_array(self, audio: np.ndarray, sample_rate: int) -> None:
        normalized = np.asarray(audio, dtype=np.float32)
        if normalized.ndim == 1:
            normalized = normalized[:, None]
        if normalized.shape[1] == 1:
            normalized = np.repeat(normalized, 2, axis=1)
        elif normalized.shape[1] > 2:
            normalized = normalized[:, :2]

        if sample_rate != self.sample_rate and normalized.size:
            normalized = self._resample(normalized, sample_rate, self.sample_rate)

        self._audio = np.clip(normalized, -1.0, 1.0).astype(np.float32)
        self.position = 0

    def play(self) -> None:
        if self.has_track:
            self.playing = True

    def pause(self) -> None:
        self.playing = False

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, volume))

    def get_chunk(self, frame_count: int) -> np.ndarray:
        if not self.playing or not self.has_track:
            return np.zeros((frame_count, 2), dtype=np.float32)

        indices = (np.arange(frame_count) + self.position) % len(self._audio)
        self.position = int((self.position + frame_count) % len(self._audio))
        return self._audio[indices] * self.volume

    @property
    def has_track(self) -> bool:
        return len(self._audio) > 0

    def _resample(
        self, audio: np.ndarray, source_rate: int, target_rate: int
    ) -> np.ndarray:
        if source_rate <= 0:
            raise ValueError("source sample rate must be positive")
        duration = len(audio) / source_rate
        target_length = max(1, int(duration * target_rate))
        source_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        target_x = np.linspace(0.0, duration, num=target_length, endpoint=False)
        channels = [
            np.interp(target_x, source_x, audio[:, channel])
            for channel in range(audio.shape[1])
        ]
        return np.stack(channels, axis=1).astype(np.float32)
