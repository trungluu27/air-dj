from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Protocol

from config.settings import (
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
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
    index_tip_x: float = 0.0
    index_tip_y: float = 0.0
    thumb_tip_x: float = 0.0
    thumb_tip_y: float = 0.0
    thumb_extended: bool = False
    index_extended: bool = False
    middle_extended: bool = False
    ring_extended: bool = False
    pinky_extended: bool = False
    palm_size: float = 0.0
    pinch_ratio: float = 1.0


class GestureDetector:
    def extract(
        self,
        label: str,
        landmarks: Iterable[LandmarkLike],
        aspect_ratio: float = 1.0,
    ) -> HandFeatures:
        """Turn one hand's landmarks into the features the mapper reads.

        ``aspect_ratio`` is the frame's width divided by its height. MediaPipe
        normalises x against the frame width and y against its height, so on the
        720x1280 portrait frame a vertical centimetre measures 1.78x larger than a
        horizontal one. Distances have to be corrected for that before they can be
        compared, or the pinch threshold would depend on how the hand is rotated.
        """
        points = list(landmarks)
        wrist = points[WRIST]
        finger_states = self._finger_states(points)
        palm_size = self._distance(points[WRIST], points[MIDDLE_MCP], aspect_ratio)
        pinch_span = self._distance(points[THUMB_TIP], points[INDEX_TIP], aspect_ratio)
        return HandFeatures(
            label=label,
            wrist_x=_clamp01(wrist.x),
            wrist_y=_clamp01(wrist.y),
            finger_count=sum(1 for is_extended in finger_states.values() if is_extended),
            index_tip_x=_clamp01(points[INDEX_TIP].x),
            index_tip_y=_clamp01(points[INDEX_TIP].y),
            thumb_tip_x=_clamp01(points[THUMB_TIP].x),
            thumb_tip_y=_clamp01(points[THUMB_TIP].y),
            thumb_extended=finger_states["thumb"],
            index_extended=finger_states["index"],
            middle_extended=finger_states["middle"],
            ring_extended=finger_states["ring"],
            pinky_extended=finger_states["pinky"],
            palm_size=palm_size,
            pinch_ratio=pinch_span / palm_size if palm_size > 1e-6 else 1.0,
        )

    def _distance(
        self, first: LandmarkLike, second: LandmarkLike, aspect_ratio: float
    ) -> float:
        aspect_ratio = aspect_ratio if aspect_ratio > 1e-6 else 1.0
        delta_x = first.x - second.x
        delta_y = (first.y - second.y) / aspect_ratio
        return math.hypot(delta_x, delta_y)

    def _count_extended_fingers(self, points: list[LandmarkLike]) -> int:
        return sum(1 for is_extended in self._finger_states(points).values() if is_extended)

    def _finger_states(self, points: list[LandmarkLike]) -> dict[str, bool]:
        states = {
            "index": points[INDEX_TIP].y < points[INDEX_PIP].y,
            "middle": points[MIDDLE_TIP].y < points[MIDDLE_PIP].y,
            "ring": points[RING_TIP].y < points[RING_PIP].y,
            "pinky": points[PINKY_TIP].y < points[PINKY_PIP].y,
        }
        # Thumb orientation varies with handedness, so use distance from wrist as
        # a simple v1 heuristic instead of left/right-specific x comparisons.
        states["thumb"] = abs(points[THUMB_TIP].x - points[WRIST].x) > 0.08
        return states


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
