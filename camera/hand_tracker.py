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
        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._drawing = mp.solutions.drawing_utils
        self._hands_module = mp.solutions.hands
        self._detector = GestureDetector()
        self._landmark_spec = self._drawing.DrawingSpec(
            color=(0, 255, 255),
            thickness=3,
            circle_radius=4,
        )
        self._connection_spec = self._drawing.DrawingSpec(
            color=(0, 180, 255),
            thickness=2,
            circle_radius=2,
        )

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
            self._drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                self._hands_module.HAND_CONNECTIONS,
                landmark_drawing_spec=self._landmark_spec,
                connection_drawing_spec=self._connection_spec,
            )
            landmarks = list(hand_landmarks.landmark)
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
            radius = 7 if is_key_point else 4
            color = (255, 80, 80) if is_key_point else (80, 255, 120)
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
