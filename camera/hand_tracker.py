from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.settings import (
    INDEX_TIP,
    MIDDLE_TIP,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
)
from gestures.gesture_detector import GestureDetector, HandFeatures


KEY_LANDMARK_LABELS = {
    WRIST: "WRIST",
    THUMB_TIP: "THUMB",
    INDEX_TIP: "INDEX",
    MIDDLE_TIP: "MIDDLE",
    RING_TIP: "RING",
    PINKY_TIP: "PINKY",
}

NEON_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class TrackedHand:
    label: str
    landmarks: list[Any]
    features: HandFeatures


@dataclass(frozen=True)
class HandResult:
    hands: list[TrackedHand]
    annotated_frame: Any


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV and MediaPipe are required for webcam tracking. "
                "Install dependencies from requirements.txt."
            ) from exc

        self._cv2 = cv2
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = GestureDetector()

    def process_rgb(self, frame: Any) -> HandResult:
        rgb_frame = frame.copy()
        results = self._hands.process(rgb_frame)
        annotated = rgb_frame.copy()
        tracked: list[TrackedHand] = []

        if not results.multi_hand_landmarks:
            return HandResult(hands=tracked, annotated_frame=annotated)

        handedness = results.multi_handedness or []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = self._label_for_index(handedness, idx)
            landmarks = list(hand_landmarks.landmark)
            self._draw_neon_hand(annotated, landmarks)
            self._draw_landmark_points(annotated, landmarks)
            tracked.append(
                TrackedHand(
                    label=label,
                    landmarks=landmarks,
                    features=self._detector.extract(label, landmarks),
                )
            )

        return HandResult(hands=tracked, annotated_frame=annotated)

    def close(self) -> None:
        self._hands.close()

    def _label_for_index(self, handedness: list[Any], idx: int) -> str:
        if idx >= len(handedness):
            return "Unknown"
        return handedness[idx].classification[0].label

    def _draw_landmark_points(self, frame: Any, landmarks: list[Any]) -> None:
        height, width = frame.shape[:2]
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            is_key_point = idx in KEY_LANDMARK_LABELS
            radius = 8 if is_key_point else 4
            color = (255, 0, 220) if is_key_point else (0, 255, 255)
            self._cv2.circle(frame, (x, y), radius + 8, color, 1)
            self._cv2.circle(frame, (x, y), radius + 3, color, 2)
            self._cv2.circle(frame, (x, y), radius, color, -1)
            self._cv2.circle(frame, (x, y), radius + 1, (255, 255, 255), 1)

            if is_key_point:
                self._cv2.putText(
                    frame,
                    KEY_LANDMARK_LABELS[idx],
                    (x + 8, y - 8),
                    self._cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    self._cv2.LINE_AA,
                )

    def _draw_neon_hand(self, frame: Any, landmarks: list[Any]) -> None:
        height, width = frame.shape[:2]
        for start_idx, end_idx in NEON_CONNECTIONS:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            p1 = (int(start.x * width), int(start.y * height))
            p2 = (int(end.x * width), int(end.y * height))
            self._cv2.line(frame, p1, p2, (255, 0, 220), 7, self._cv2.LINE_AA)
            self._cv2.line(frame, p1, p2, (0, 255, 255), 3, self._cv2.LINE_AA)
            self._cv2.line(frame, p1, p2, (255, 255, 255), 1, self._cv2.LINE_AA)
