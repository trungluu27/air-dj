from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from config.settings import DEFAULT_SETTINGS
from core.session import DJSession
from gestures.gesture_detector import HandFeatures
from gestures.pinch import PinchDetector, PinchEvent, PinchState

# The middle of the frame is filter-off; a hand has to travel a little way out of
# it before the sweep engages, so a hold that is only meant to cancel a transition
# does not also open the filter.
FILTER_DEADZONE = 0.08


@dataclass
class GestureMapper:
    """Maps one gesture - a pinch - onto the mixer.

    Everything else was removed on purpose. Volume followed hand height, which
    meant a hand could never rest; the crossfader needed both hands in frame at
    once; and play, pause, tempo sync and beat match were all read off a finger
    count that flipped whenever a hand tilted. Four meanings on one noisy signal
    is what made the old mapping so hard to control.

    What is left:

    * pinch tap    - ask for the next automatic transition
    * pinch hold, hand moved up or down - sweep the master filter. Moving away
                     from the centre is what marks this as a sweep rather than a
                     mix, and only then is the transition request withdrawn.
                     Holding still is harmless, so an ordinary slow pinch still
                     mixes.
    * hand leaves  - the filter returns to centre on its own

    Only ever one hand at a time: the first one seen is locked and keeps
    control until it leaves the frame.

    The mapper never touches audio timing. It only raises a request; the
    transition engine decides when that becomes audible.
    """

    smoothing_alpha: float = DEFAULT_SETTINGS.filter_smoothing_alpha
    hand_release_seconds: float = DEFAULT_SETTINGS.hand_release_seconds
    detector: PinchDetector = field(default_factory=PinchDetector)
    _sweeping: bool = False
    _locked_label: str | None = None
    _locked_seen_at: float = 0.0

    def apply(
        self,
        session: DJSession,
        hands: Iterable[HandFeatures],
        now: float | None = None,
    ) -> DJSession:
        current_time = time.monotonic() if now is None else now
        by_label = {hand.label: hand for hand in hands}

        hand, lock_held = self._locked_hand(by_label, current_time)
        session.active_hand = self._locked_label or ""
        if hand is None:
            if lock_held:
                # Tracking dropped the hand for a frame or two. Freeze rather than
                # letting go of a filter sweep over a blink.
                return session
            # Nothing to hold the filter open, so let it fall back to centre.
            self._sweeping = False
            session.pinch_state = "no hand"
            session.filter_master = self._smooth(session.filter_master, 0.0)
            return session

        event = self.detector.update(hand, current_time)
        session.pinch_ratio = event.ratio

        if event.closed_edge:
            session.automix_requested = True
            self._sweeping = False
        if event.state is PinchState.OPEN:
            self._sweeping = False

        target = self._filter_target(event)
        if target != 0.0 and not self._sweeping:
            # The hand has been held pinched *and* moved away from the centre, so
            # this is a filter sweep rather than a mix. Only now is the transition
            # request withdrawn - the engine is still waiting for a downbeat, so
            # nothing has been heard yet.
            #
            # Holding still used to be enough to cancel, which made an ordinary
            # pinch lasting longer than a third of a second do nothing at all, with
            # no feedback to say why.
            self._sweeping = True
            session.automix_requested = False
            session.automix_cancel_requested = True

        session.filter_master = self._smooth(session.filter_master, target)
        # Labelled last, so the readout reflects the sweep decision taken just
        # above rather than trailing it by a frame.
        session.pinch_state = self._pinch_label(event)
        return session

    def _pinch_label(self, event: PinchEvent) -> str:
        if event.state is PinchState.OPEN:
            return "open"
        if self._sweeping:
            return "sweeping"
        return "hold" if event.holding else "pinched"

    def _locked_hand(
        self, by_label: dict[str, HandFeatures], now: float
    ) -> tuple[HandFeatures | None, bool]:
        """Follow one hand and ignore every other.

        Control used to go to whichever hand the right-hand-wins rule picked each
        frame, so a second hand drifting into view took over mid-gesture. Now the
        first hand to arrive is locked and keeps control until it leaves; anything
        else in frame is not looked at.

        Returns the hand to read and whether a lock is still being held. A held
        lock with no hand means tracking blinked, and the caller should freeze
        rather than reset.
        """
        if self._locked_label is not None:
            hand = by_label.get(self._locked_label)
            if hand is not None:
                self._locked_seen_at = now
                return hand, True
            if now - self._locked_seen_at < self.hand_release_seconds:
                return None, True
            self._release_lock()

        if not by_label:
            return None, False

        # Nothing locked yet. If both hands arrive on the same frame the right one
        # takes it, otherwise there is only one to choose.
        for label in ("Right", "Left"):
            if label in by_label:
                self._locked_label = label
                break
        else:
            self._locked_label = next(iter(by_label))
        self._locked_seen_at = now
        return by_label[self._locked_label], True

    def _release_lock(self) -> None:
        if self._locked_label is not None:
            self.detector.release(self._locked_label)
        self._locked_label = None
        self._sweeping = False

    def _filter_target(self, event: PinchEvent) -> float:
        if not event.holding:
            return 0.0
        # wrist_y is 0 at the top of the frame: raising the hand opens a high-pass
        # and thins the track out, lowering it closes a low-pass and muffles it.
        offset = 0.5 - event.hold_y
        if abs(offset) < FILTER_DEADZONE:
            return 0.0
        span = max(1e-6, 0.5 - FILTER_DEADZONE)
        sign = 1.0 if offset > 0.0 else -1.0
        return sign * min(1.0, (abs(offset) - FILTER_DEADZONE) / span)

    def _smooth(self, previous: float, new_value: float) -> float:
        alpha = max(0.0, min(1.0, self.smoothing_alpha))
        return max(-1.0, min(1.0, alpha * new_value + (1.0 - alpha) * previous))
