from __future__ import annotations

from dataclasses import dataclass, field

from gestures.gesture_classifier import GestureType


@dataclass
class DJSession:
    volume_a: float = 0.75
    volume_b: float = 0.75
    crossfader: float = 0.5
    playing_a: bool = False
    playing_b: bool = False
    bpm: float = 120.0
    effects: set[str] = field(default_factory=set)
    last_gesture_a: GestureType = GestureType.UNKNOWN
    last_gesture_b: GestureType = GestureType.UNKNOWN

    def as_dict(self) -> dict[str, object]:
        return {
            "volume_a": round(self.volume_a, 3),
            "volume_b": round(self.volume_b, 3),
            "crossfader": round(self.crossfader, 3),
            "playing_a": self.playing_a,
            "playing_b": self.playing_b,
            "bpm": round(self.bpm, 1),
            "effects": sorted(self.effects),
            "last_gesture_a": self.last_gesture_a.value,
            "last_gesture_b": self.last_gesture_b.value,
        }
