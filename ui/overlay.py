from __future__ import annotations

from typing import Any, Iterable

from core.session import DJSession
from gestures.gesture_detector import HandFeatures


class Overlay:
    def __init__(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for drawing the AirDJ overlay."
            ) from exc
        self._cv2 = cv2

    def draw(
        self,
        frame: Any,
        session: DJSession,
        hands: Iterable[HandFeatures],
        status: str = "Ready",
    ) -> Any:
        output = frame.copy()
        self._draw_header(output, status)
        self._draw_deck(output, "Deck A", session.volume_a, session.playing_a, 24, 72)
        self._draw_deck(output, "Deck B", session.volume_b, session.playing_b, 24, 152)
        self._draw_crossfader(output, session.crossfader, 24, 236)
        self._draw_hands(output, hands, 24, 306)
        return output

    def _draw_header(self, frame: Any, status: str) -> None:
        self._cv2.putText(
            frame,
            f"AirDJ - {status}",
            (24, 36),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            self._cv2.LINE_AA,
        )

    def _draw_deck(
        self, frame: Any, name: str, volume: float, playing: bool, x: int, y: int
    ) -> None:
        state = "PLAYING" if playing else "PAUSED"
        self._cv2.putText(
            frame,
            f"{name}: {state}",
            (x, y),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            self._cv2.LINE_AA,
        )
        self._draw_bar(frame, x, y + 18, 220, 18, volume)
        self._cv2.putText(
            frame,
            f"vol {int(volume * 100):3d}%",
            (x + 235, y + 34),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 255, 200),
            1,
            self._cv2.LINE_AA,
        )

    def _draw_crossfader(self, frame: Any, value: float, x: int, y: int) -> None:
        self._cv2.putText(
            frame,
            "Crossfader",
            (x, y),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            self._cv2.LINE_AA,
        )
        self._draw_bar(frame, x, y + 18, 300, 16, value)

    def _draw_hands(
        self, frame: Any, hands: Iterable[HandFeatures], x: int, y: int
    ) -> None:
        offset = 0
        for hand in hands:
            self._cv2.putText(
                frame,
                f"{hand.label}: fingers={hand.finger_count} wrist=({hand.wrist_x:.2f}, {hand.wrist_y:.2f})",
                (x, y + offset),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                self._cv2.LINE_AA,
            )
            offset += 24

    def _draw_bar(
        self, frame: Any, x: int, y: int, width: int, height: int, value: float
    ) -> None:
        value = max(0.0, min(1.0, value))
        self._cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 1)
        self._cv2.rectangle(
            frame,
            (x, y),
            (x + int(width * value), y + height),
            (0, 220, 120),
            -1,
        )
