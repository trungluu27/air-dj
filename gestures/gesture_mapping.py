from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from config.settings import DEFAULT_SETTINGS
from core.session import DJSession
from gestures.gesture_classifier import GestureClassifier, GestureType
from gestures.gesture_detector import HandFeatures


@dataclass
class GestureMapper:
    smoothing_alpha: float = DEFAULT_SETTINGS.smoothing_alpha
    play_pause_cooldown_seconds: float = DEFAULT_SETTINGS.play_pause_cooldown_seconds
    classifier: GestureClassifier = field(default_factory=GestureClassifier)
    _last_play_pause_at: dict[str, float] = field(
        default_factory=lambda: {"Left": -999.0, "Right": -999.0}
    )

    def apply(
        self,
        session: DJSession,
        hands: Iterable[HandFeatures],
        now: float | None = None,
    ) -> DJSession:
        current_time = time.monotonic() if now is None else now
        by_label = {hand.label: hand for hand in hands}

        left = by_label.get("Left")
        right = by_label.get("Right")

        if left is not None:
            session.volume_a = self._smooth(session.volume_a, 1.0 - left.wrist_y)
            self._apply_play_pause(session, left, current_time)

        if right is not None:
            session.volume_b = self._smooth(session.volume_b, 1.0 - right.wrist_y)
            self._apply_play_pause(session, right, current_time)

        if left is not None and right is not None:
            midpoint = (left.wrist_x + right.wrist_x) / 2.0
            session.crossfader = self._smooth(session.crossfader, midpoint)

        return session

    def _apply_play_pause(
        self, session: DJSession, hand: HandFeatures, now: float
    ) -> None:
        gesture = self.classifier.classify(hand.finger_count)
        if gesture is GestureType.UNKNOWN:
            return

        if now - self._last_play_pause_at[hand.label] < self.play_pause_cooldown_seconds:
            return

        if hand.label == "Left":
            session.playing_a = gesture is GestureType.OPEN_PALM
            session.last_gesture_a = gesture
        elif hand.label == "Right":
            session.playing_b = gesture is GestureType.OPEN_PALM
            session.last_gesture_b = gesture
        else:
            return

        self._last_play_pause_at[hand.label] = now

    def _smooth(self, previous: float, new_value: float) -> float:
        alpha = max(0.0, min(1.0, self.smoothing_alpha))
        smoothed = alpha * max(0.0, min(1.0, new_value)) + (1.0 - alpha) * previous
        return max(0.0, min(1.0, smoothed))
