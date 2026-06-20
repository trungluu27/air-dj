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
        hands_list = list(hands)
        self._draw_vignette(output)
        self._draw_header(output, status)
        self._draw_vertical_deck(output, "DECK A", session.volume_a, session.playing_a, 24)
        self._draw_vertical_deck(
            output,
            "DECK B",
            session.volume_b,
            session.playing_b,
            output.shape[1] - 82,
        )
        self._draw_crossfader(output, session.crossfader)
        self._draw_hands(output, hands_list)
        return output

    def _draw_vignette(self, frame: Any) -> None:
        overlay = frame.copy()
        height, width = frame.shape[:2]
        self._cv2.rectangle(overlay, (0, 0), (width, 120), (16, 8, 32), -1)
        self._cv2.rectangle(overlay, (0, height - 190), (width, height), (16, 8, 32), -1)
        self._cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    def _draw_header(self, frame: Any, status: str) -> None:
        width = frame.shape[1]
        self._cv2.putText(
            frame,
            "AIRDJ",
            (28, 58),
            self._cv2.FONT_HERSHEY_DUPLEX,
            1.45,
            (0, 255, 255),
            2,
            self._cv2.LINE_AA,
        )
        self._cv2.putText(
            frame,
            "GESTURE MIX MODE",
            (32, 88),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 0, 210),
            1,
            self._cv2.LINE_AA,
        )
        self._cv2.putText(
            frame,
            status.upper(),
            (width - 190, 58),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 255, 255),
            1,
            self._cv2.LINE_AA,
        )

    def _draw_vertical_deck(
        self, frame: Any, name: str, volume: float, playing: bool, x: int
    ) -> None:
        height = frame.shape[0]
        meter_top = 190
        meter_height = height - 430
        meter_width = 42
        state = "PLAYING" if playing else "PAUSED"
        color = (0, 255, 255) if playing else (255, 0, 210)

        self._cv2.rectangle(
            frame,
            (x, meter_top),
            (x + meter_width, meter_top + meter_height),
            (70, 40, 92),
            2,
        )
        fill_height = int(meter_height * max(0.0, min(1.0, volume)))
        y1 = meter_top + meter_height - fill_height
        self._cv2.rectangle(
            frame,
            (x + 6, y1),
            (x + meter_width - 6, meter_top + meter_height - 6),
            color,
            -1,
        )
        self._cv2.rectangle(
            frame,
            (x + 6, y1),
            (x + meter_width - 6, meter_top + meter_height - 6),
            (255, 255, 255),
            1,
        )
        self._cv2.putText(
            frame,
            name,
            (x - 4, meter_top - 24),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            self._cv2.LINE_AA,
        )
        self._cv2.putText(
            frame,
            f"{int(volume * 100):02d}%",
            (x - 2, meter_top + meter_height + 32),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            self._cv2.LINE_AA,
        )
        self._draw_badge(frame, state, x - 6, meter_top + meter_height + 48, color)

    def _draw_crossfader(self, frame: Any, value: float) -> None:
        height, width = frame.shape[:2]
        x = 120
        y = height - 130
        bar_width = width - 240
        value = max(0.0, min(1.0, value))
        self._cv2.putText(
            frame,
            "CROSSFADER",
            (x, y),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 255, 255),
            1,
            self._cv2.LINE_AA,
        )
        self._cv2.line(frame, (x, y + 36), (x + bar_width, y + 36), (70, 40, 92), 10)
        knob_x = x + int(bar_width * value)
        self._cv2.circle(frame, (knob_x, y + 36), 17, (255, 0, 210), -1)
        self._cv2.circle(frame, (knob_x, y + 36), 23, (0, 255, 255), 2)

    def _draw_hands(
        self, frame: Any, hands: Iterable[HandFeatures]
    ) -> None:
        x = 28
        y = 120
        offset = 0
        for hand in hands:
            color = (0, 255, 255) if hand.label == "Left" else (255, 0, 210)
            self._cv2.putText(
                frame,
                f"{hand.label.upper()}  FINGERS:{hand.finger_count}  X:{hand.wrist_x:.2f} Y:{hand.wrist_y:.2f}",
                (x, y + offset),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                self._cv2.LINE_AA,
            )
            offset += 24

    def _draw_badge(self, frame: Any, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        text_width = max(82, len(text) * 10)
        self._cv2.rectangle(frame, (x, y), (x + text_width, y + 24), (16, 8, 32), -1)
        self._cv2.rectangle(frame, (x, y), (x + text_width, y + 24), color, 1)
        self._cv2.putText(
            frame,
            text,
            (x + 8, y + 17),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            self._cv2.LINE_AA,
        )

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
