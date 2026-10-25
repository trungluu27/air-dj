from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from core.session import DJSession
from gestures.gesture_detector import HandFeatures

if TYPE_CHECKING:
    from audio.deck import Deck


def _waveform_viewport(progress: float, visible_fraction: float = 0.18) -> tuple[float, float]:
    visible_fraction = max(0.05, min(1.0, visible_fraction))
    progress = max(0.0, min(1.0, progress))
    half_window = visible_fraction / 2.0
    start = progress - half_window
    end = progress + half_window
    if start < 0.0:
        start = 0.0
        end = visible_fraction
    elif end > 1.0:
        end = 1.0
        start = 1.0 - visible_fraction
    return round(start, 6), round(end, 6)


def _visible_beat_offsets(beat_positions: Any, start: float, end: float) -> Any:
    import numpy as np

    if end <= start:
        return np.zeros(0, dtype=np.float32)
    beats = np.asarray(beat_positions, dtype=np.float32)
    visible = beats[(beats >= start) & (beats <= end)]
    return ((visible - start) / (end - start)).astype(np.float32)


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
        deck_a: Deck | None = None,
        deck_b: Deck | None = None,
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
        self._draw_beatmatch_display(output, deck_a, deck_b, 96, 690, output.shape[1] - 192)
        self._draw_crossfader(output, session.crossfader)
        self._draw_transition(output, session)
        self._draw_pinch_meter(output, session)
        self._draw_filter(output, session.filter_master)
        self._draw_hands(output, hands_list)
        return output

    def _draw_transition(self, frame: Any, session: DJSession) -> None:
        """Show what the automix is doing.

        The whole point of the redesign is that the machine decides when a mix
        happens, so the user has to be able to see it coming and see it land -
        otherwise a gesture that does nothing for a second feels broken.
        """
        height, width = frame.shape[:2]
        x = 120
        y = height - 62
        bar_width = width - 240
        phase = session.transition_phase

        if phase == "armed":
            label = f"CUEING {session.transition_label} - waiting for the downbeat"
            color = (0, 210, 255)
        elif phase == "running":
            beats = session.transition_beats_remaining
            label = f"MIXING {session.transition_label} - {beats:.0f} beats left"
            color = (120, 255, 140)
        else:
            label = session.transition_status.upper()
            color = (150, 140, 170)

        self._cv2.putText(
            frame,
            label,
            (x, y),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            self._cv2.LINE_AA,
        )
        self._cv2.line(frame, (x, y + 20), (x + bar_width, y + 20), (54, 32, 72), 8)
        if phase == "running":
            filled = int(bar_width * max(0.0, min(1.0, session.transition_progress)))
            self._cv2.line(frame, (x, y + 20), (x + filled, y + 20), color, 8)
            # The halfway mark is where the bass hands over, which is the moment
            # the mix actually commits.
            swap_x = x + bar_width // 2
            self._cv2.line(frame, (swap_x, y + 12), (swap_x, y + 28), (255, 255, 255), 2)
        elif phase == "armed":
            self._cv2.circle(frame, (x + 4, y + 20), 7, color, -1)

    def _draw_pinch_meter(self, frame: Any, session: DJSession) -> None:
        """Show how closed the pinch is, and where the trigger sits.

        Without this the gesture is unteachable: a pinch that does not quite cross
        the threshold looks exactly like no gesture at all, and there is no way to
        tell whether the camera sees your hand, whether you are pinching hard
        enough, or which of the two actions you just performed.
        """
        from config.settings import DEFAULT_SETTINGS

        height, width = frame.shape[:2]
        x = width - 150
        y = 150
        bar_height = 190
        bar_width = 16

        state = session.pinch_state
        colors = {
            "no hand": (120, 110, 135),
            "open": (170, 200, 215),
            "pinched": (120, 255, 140),
            "hold": (0, 210, 255),
            "sweeping": (255, 210, 0),
        }
        color = colors.get(state, (170, 200, 215))

        self._cv2.putText(
            frame,
            "PINCH",
            (x - 6, y - 14),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (180, 255, 255),
            1,
            self._cv2.LINE_AA,
        )
        self._cv2.rectangle(
            frame, (x, y), (x + bar_width, y + bar_height), (54, 32, 72), -1
        )

        # The bar fills as the fingers close, so it grows towards the trigger.
        closed = max(0.0, min(1.0, 1.0 - min(1.0, session.pinch_ratio)))
        filled = int(bar_height * closed)
        if state != "no hand":
            self._cv2.rectangle(
                frame,
                (x, y + bar_height - filled),
                (x + bar_width, y + bar_height),
                color,
                -1,
            )

        trigger = 1.0 - DEFAULT_SETTINGS.pinch_close_ratio
        trigger_y = y + bar_height - int(bar_height * trigger)
        self._cv2.line(
            frame, (x - 7, trigger_y), (x + bar_width + 7, trigger_y), (255, 0, 210), 2
        )
        self._cv2.putText(
            frame,
            "FIRE",
            (x - 46, trigger_y + 5),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 0, 210),
            1,
            self._cv2.LINE_AA,
        )
        self._cv2.putText(
            frame,
            state.upper(),
            (x - 30, y + bar_height + 22),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1,
            self._cv2.LINE_AA,
        )

    def _draw_filter(self, frame: Any, value: float) -> None:
        if abs(value) < 0.02:
            return
        height, width = frame.shape[:2]
        centre_x = width // 2
        y = height - 175
        half = (width - 300) // 2
        travel = int(half * max(-1.0, min(1.0, value)))
        name = "HIGH PASS" if value > 0.0 else "LOW PASS"
        self._cv2.line(frame, (centre_x - half, y), (centre_x + half, y), (54, 32, 72), 6)
        self._cv2.line(frame, (centre_x, y), (centre_x + travel, y), (255, 210, 0), 6)
        self._cv2.putText(
            frame,
            f"{name} {abs(value) * 100:.0f}%",
            (centre_x - 70, y - 12),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 210, 0),
            1,
            self._cv2.LINE_AA,
        )

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
        y = height - 148
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

    def _draw_beatmatch_display(
        self,
        frame: Any,
        deck_a: Deck | None,
        deck_b: Deck | None,
        x: int,
        y: int,
        width: int,
    ) -> None:
        panel_height = 250
        self._cv2.rectangle(frame, (x, y), (x + width, y + panel_height), (10, 14, 24), -1)
        self._cv2.rectangle(frame, (x, y), (x + width, y + panel_height), (70, 210, 220), 1)
        self._cv2.putText(
            frame,
            "BEATMATCH WAVEFORMS",
            (x + 14, y + 28),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 255, 255),
            1,
            self._cv2.LINE_AA,
        )

        graph_x = x + 80
        graph_width = width - 104
        lane_a_y = y + 74
        lane_b_y = y + 164
        lane_height = 62

        self._draw_waveform_lane(
            frame,
            "A",
            deck_a,
            graph_x,
            lane_a_y,
            graph_width,
            lane_height,
            (0, 255, 255),
        )
        self._draw_waveform_lane(
            frame,
            "B",
            deck_b,
            graph_x,
            lane_b_y,
            graph_width,
            lane_height,
            (80, 255, 120),
        )

        playhead_x = graph_x + graph_width // 2
        self._cv2.line(frame, (playhead_x, y + 48), (playhead_x, y + panel_height - 14), (255, 255, 255), 2)
        self._cv2.circle(frame, (playhead_x, y + 46), 5, (255, 255, 255), -1)

    def _draw_waveform_lane(
        self,
        frame: Any,
        label: str,
        deck: Deck | None,
        x: int,
        y: int,
        width: int,
        height: int,
        color: tuple[int, int, int],
    ) -> None:
        center_y = y + height // 2
        self._cv2.rectangle(frame, (x, y), (x + width, y + height), (18, 28, 38), -1)
        for grid_x in range(x, x + width + 1, max(1, width // 16)):
            self._cv2.line(frame, (grid_x, y), (grid_x, y + height), (38, 58, 68), 1)
        self._cv2.line(frame, (x, center_y), (x + width, center_y), (75, 105, 118), 1)

        track_name = "No track"
        bpm_text = "-- BPM"
        tempo_text = "+0.0%"
        peaks = []
        beat_offsets = []
        viewport_start = 0.0
        viewport_end = 1.0
        progress_offset = 0.5
        if deck is not None and deck.has_track:
            track_name = deck.track_name[:18]
            bpm_text = f"{deck.effective_bpm:.0f} BPM" if deck.effective_bpm else "-- BPM"
            tempo_text = f"{(deck.tempo_ratio - 1.0) * 100:+.1f}%"
            peaks = deck.waveform_peaks
            viewport_start, viewport_end = _waveform_viewport(deck.progress)
            beat_offsets = _visible_beat_offsets(deck.beat_positions, viewport_start, viewport_end)
            progress_offset = (deck.progress - viewport_start) / max(0.001, viewport_end - viewport_start)

        self._cv2.putText(
            frame,
            f"DECK {label}",
            (x - 70, y + 24),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            self._cv2.LINE_AA,
        )
        self._cv2.putText(
            frame,
            f"{bpm_text} {tempo_text}",
            (x - 76, y + 48),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (230, 240, 255),
            1,
            self._cv2.LINE_AA,
        )
        self._cv2.putText(
            frame,
            track_name,
            (x + 8, y - 8),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (210, 230, 235),
            1,
            self._cv2.LINE_AA,
        )

        for beat_offset in beat_offsets:
            beat_x = x + int(width * float(beat_offset))
            self._cv2.line(frame, (beat_x, y - 7), (beat_x, y + height + 7), (235, 245, 245), 1)
            self._cv2.circle(frame, (beat_x, y + height + 8), 3, color, -1)

        if len(peaks):
            for offset_x in range(width):
                sample_ratio = offset_x / max(1, width - 1)
                # Show only the local viewport around the playhead, like DJ beatmatch lanes.
                peak_position = viewport_start + sample_ratio * (viewport_end - viewport_start)
                peak_index = int(peak_position * (len(peaks) - 1))
                amplitude = int(float(peaks[peak_index]) * height * 0.42)
                px = x + offset_x
                self._cv2.line(frame, (px, center_y - amplitude), (px, center_y + amplitude), color, 1)

        playhead_x = x + int(width * max(0.0, min(1.0, progress_offset)))
        self._cv2.line(frame, (playhead_x, y - 8), (playhead_x, y + height + 8), (255, 255, 255), 1)

    def _draw_deck_waveform(
        self,
        frame: Any,
        name: str,
        deck: Deck | None,
        x: int,
        y: int,
        width: int,
        color: tuple[int, int, int],
    ) -> None:
        panel_height = 104
        self._cv2.rectangle(frame, (x, y), (x + width, y + panel_height), (16, 8, 32), -1)
        self._cv2.rectangle(frame, (x, y), (x + width, y + panel_height), color, 1)

        track_name = "No track"
        bpm_text = "-- BPM"
        tempo_text = f"{1.0:.2f}x"
        peaks = []
        beat_positions = []
        progress = 0.0
        if deck is not None and deck.has_track:
            track_name = deck.track_name[:34]
            bpm_text = f"{deck.effective_bpm:.0f} BPM" if deck.effective_bpm else "-- BPM"
            tempo_text = f"{(deck.tempo_ratio - 1.0) * 100:+.1f}%"
            peaks = deck.waveform_peaks
            beat_positions = deck.beat_positions
            progress = deck.progress

        self._cv2.putText(
            frame,
            f"{name}  {track_name}",
            (x + 12, y + 24),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            self._cv2.LINE_AA,
        )
        self._cv2.putText(
            frame,
            f"{bpm_text}  TEMPO {tempo_text}",
            (x + width - 174, y + 24),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            self._cv2.LINE_AA,
        )

        wave_top = y + 42
        wave_height = 46
        center_y = wave_top + wave_height // 2
        wave_width = width - 24
        self._cv2.line(frame, (x + 12, center_y), (x + width - 12, center_y), (70, 40, 92), 1)
        for beat_position in beat_positions:
            if 0.0 <= float(beat_position) <= 1.0:
                beat_x = x + 12 + int(wave_width * float(beat_position))
                self._cv2.line(
                    frame,
                    (beat_x, wave_top - 4),
                    (beat_x, wave_top + wave_height + 4),
                    (180, 255, 255),
                    1,
                )

        if len(peaks):
            for offset_x in range(wave_width):
                peak_index = int(offset_x / max(1, wave_width - 1) * (len(peaks) - 1))
                amplitude = int(float(peaks[peak_index]) * wave_height / 2)
                px = x + 12 + offset_x
                self._cv2.line(frame, (px, center_y - amplitude), (px, center_y + amplitude), color, 1)

        playhead_x = x + 12 + int((width - 24) * max(0.0, min(1.0, progress)))
        self._cv2.line(frame, (playhead_x, wave_top - 3), (playhead_x, wave_top + wave_height + 3), (255, 255, 255), 2)

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
