from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from config.settings import DEFAULT_SETTINGS
from gestures.gesture_detector import HandFeatures


class PinchState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True)
class PinchEvent:
    """What one hand did this frame.

    ``closed_edge`` fires on the single frame the pinch closes. Waiting for the
    release instead would be more natural, but the video loop only samples a few
    times a second and a quick tap can happen entirely between two frames, so the
    trigger has to be the closing edge.

    ``hold_started`` fires once, on the frame a pinch has been held long enough to
    count as a hold rather than a tap, which is the caller's chance to undo
    whatever the closing edge set off. ``holding`` then stays true for the
    duration and carries the hand position so a hold can drive a continuous
    control.
    """

    label: str
    state: PinchState = PinchState.OPEN
    closed_edge: bool = False
    hold_started: bool = False
    holding: bool = False
    released: bool = False
    held_for: float = 0.0
    hold_x: float = 0.0
    hold_y: float = 0.0
    anchor_x: float = 0.0
    anchor_y: float = 0.0
    ratio: float = 1.0

    @property
    def travel_y(self) -> float:
        """How far the hand has moved since the pinch closed, downwards positive.

        A hold has to be measured against where it started, not against the middle
        of the frame: a hand resting naturally at chest height is nowhere near the
        centre, and treating that offset as a deliberate move made holding still
        read as a gesture.
        """
        return self.hold_y - self.anchor_y


@dataclass
class PinchDetector:
    """Turns thumb-to-index distance into a tap or a hold.

    This replaces finger counting, which compared each fingertip against its own
    knuckle and so flipped meaning the moment a hand tilted. A pinch is a single
    ratio against palm size: independent of how far the hand is from the camera,
    independent of how it is rotated, and with separate close and open thresholds
    so a ratio resting on the boundary cannot chatter.
    """

    close_ratio: float = DEFAULT_SETTINGS.pinch_close_ratio
    open_ratio: float = DEFAULT_SETTINGS.pinch_open_ratio
    tap_max_seconds: float = DEFAULT_SETTINGS.pinch_tap_max_seconds
    rearm_seconds: float = DEFAULT_SETTINGS.pinch_rearm_seconds
    _state: dict[str, PinchState] = field(default_factory=dict)
    _closed_at: dict[str, float] = field(default_factory=dict)
    _promoted: dict[str, bool] = field(default_factory=dict)
    _last_close_at: dict[str, float] = field(default_factory=dict)
    _anchor: dict[str, tuple[float, float]] = field(default_factory=dict)

    def update(self, hand: HandFeatures, now: float) -> PinchEvent:
        label = hand.label
        ratio = hand.pinch_ratio

        if self._state.get(label, PinchState.OPEN) is PinchState.OPEN:
            if ratio >= self.close_ratio:
                return PinchEvent(label=label, ratio=ratio)
            if now - self._last_close_at.get(label, -999.0) < self.rearm_seconds:
                # Still inside the debounce window from the last pinch; track the
                # state so a genuine hold is not mistaken for a new tap later.
                self._state[label] = PinchState.CLOSED
                self._closed_at[label] = now
                self._promoted[label] = True
                self._anchor[label] = (hand.wrist_x, hand.wrist_y)
                return PinchEvent(label=label, state=PinchState.CLOSED, ratio=ratio)
            self._state[label] = PinchState.CLOSED
            self._closed_at[label] = now
            self._promoted[label] = False
            self._last_close_at[label] = now
            self._anchor[label] = (hand.wrist_x, hand.wrist_y)
            return PinchEvent(
                label=label,
                state=PinchState.CLOSED,
                closed_edge=True,
                hold_x=hand.wrist_x,
                hold_y=hand.wrist_y,
                anchor_x=hand.wrist_x,
                anchor_y=hand.wrist_y,
                ratio=ratio,
            )

        # Currently closed: only the wider open threshold releases it.
        held_for = now - self._closed_at.get(label, now)
        if ratio <= self.open_ratio:
            is_hold = held_for >= self.tap_max_seconds
            hold_started = is_hold and not self._promoted.get(label, False)
            if hold_started:
                self._promoted[label] = True
            anchor_x, anchor_y = self._anchor.get(
                label, (hand.wrist_x, hand.wrist_y)
            )
            return PinchEvent(
                label=label,
                state=PinchState.CLOSED,
                hold_started=hold_started,
                holding=is_hold,
                held_for=held_for,
                hold_x=hand.wrist_x,
                hold_y=hand.wrist_y,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                ratio=ratio,
            )

        self._state[label] = PinchState.OPEN
        self._promoted[label] = False
        return PinchEvent(
            label=label,
            state=PinchState.OPEN,
            released=True,
            held_for=held_for,
            hold_x=hand.wrist_x,
            hold_y=hand.wrist_y,
            ratio=ratio,
        )

    def release(self, label: str) -> PinchEvent:
        """Forget a hand that has left the frame.

        Returns a release event so the caller can let go of whatever the hand was
        holding, rather than leaving a filter stuck open.
        """
        was_closed = self._state.pop(label, PinchState.OPEN) is PinchState.CLOSED
        self._closed_at.pop(label, None)
        self._promoted.pop(label, None)
        self._anchor.pop(label, None)
        return PinchEvent(label=label, released=was_closed)

    def tracked_labels(self) -> list[str]:
        return list(self._state)
