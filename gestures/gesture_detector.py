from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from config.settings import (
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
)


class LandmarkLike(Protocol):
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class HandFeatures:
    label: str
    wrist_x: float
    wrist_y: float
    finger_count: int


class GestureDetector:
    def extract(self, label: str, landmarks: Iterable[LandmarkLike]) -> HandFeatures:
        points = list(landmarks)
        wrist = points[WRIST]
        return HandFeatures(
            label=label,
            wrist_x=_clamp01(wrist.x),
            wrist_y=_clamp01(wrist.y),
            finger_count=self._count_extended_fingers(points),
        )

    def _count_extended_fingers(self, points: list[LandmarkLike]) -> int:
        count = 0
        for tip_idx, pip_idx in (
            (INDEX_TIP, INDEX_PIP),
            (MIDDLE_TIP, MIDDLE_PIP),
            (RING_TIP, RING_PIP),
            (PINKY_TIP, PINKY_PIP),
        ):
            if points[tip_idx].y < points[pip_idx].y:
                count += 1

        # Thumb orientation varies with handedness, so use distance from wrist as
        # a simple v1 heuristic instead of left/right-specific x comparisons.
        if abs(points[THUMB_TIP].x - points[WRIST].x) > 0.08:
            count += 1

        return count


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
