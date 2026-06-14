from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

import numpy as np

from audio.deck import Deck
from config.settings import DEFAULT_SETTINGS
from core.session import DJSession


@dataclass
class Mixer:
    sample_rate: int = DEFAULT_SETTINGS.sample_rate
    deck_a: Deck | None = None
    deck_b: Deck | None = None
    crossfader: float = 0.5
    _lock: RLock = field(default_factory=RLock, repr=False)

    def __post_init__(self) -> None:
        if self.deck_a is None:
            self.deck_a = Deck("A", self.sample_rate)
        if self.deck_b is None:
            self.deck_b = Deck("B", self.sample_rate)

    def apply_session(self, session: DJSession) -> None:
        with self._lock:
            assert self.deck_a is not None
            assert self.deck_b is not None
            self.deck_a.set_volume(session.volume_a)
            self.deck_b.set_volume(session.volume_b)
            self.deck_a.playing = session.playing_a and self.deck_a.has_track
            self.deck_b.playing = session.playing_b and self.deck_b.has_track
            self.crossfader = max(0.0, min(1.0, session.crossfader))

    def get_chunk(self, frame_count: int) -> np.ndarray:
        with self._lock:
            assert self.deck_a is not None
            assert self.deck_b is not None
            left_gain = 1.0 - self.crossfader
            right_gain = self.crossfader
            mixed = (
                self.deck_a.get_chunk(frame_count) * left_gain
                + self.deck_b.get_chunk(frame_count) * right_gain
            )
            return np.clip(mixed, -1.0, 1.0).astype(np.float32)

    def load_deck(self, deck_name: str, path: str | Path) -> None:
        with self._lock:
            self._deck(deck_name).load(path)

    def load_deck_array(
        self, deck_name: str, audio: np.ndarray, sample_rate: int
    ) -> None:
        with self._lock:
            self._deck(deck_name).load_array(audio, sample_rate)

    def _deck(self, deck_name: str) -> Deck:
        assert self.deck_a is not None
        assert self.deck_b is not None
        if deck_name == "A":
            return self.deck_a
        if deck_name == "B":
            return self.deck_b
        raise ValueError(f"Unknown deck: {deck_name}")
