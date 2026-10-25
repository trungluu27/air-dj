from __future__ import annotations

from dataclasses import dataclass, field

from gestures.gesture_classifier import GestureType


@dataclass
class DJSession:
    volume_a: float = 0.75
    volume_b: float = 0.75
    crossfader: float = 0.5
    playing_a: bool = False
    playing_b: bool = False
    bpm: float = 120.0
    tempo_ratio_a: float = 1.0
    tempo_ratio_b: float = 1.0
    nudge_a: float = 0.0
    nudge_b: float = 0.0
    sync_tempo_requested: bool = False
    beat_match_requested: bool = False
    sync_status: str = "manual"
    automix_requested: bool = False
    automix_cancel_requested: bool = False
    automix_beats: int = 16
    transition_phase: str = "idle"
    transition_progress: float = 0.0
    transition_beats_remaining: float = 0.0
    transition_label: str = ""
    transition_status: str = "ready"
    filter_master: float = 0.0
    pinch_state: str = "no hand"
    pinch_ratio: float = 1.0
    effects: set[str] = field(default_factory=set)
    last_gesture_a: GestureType = GestureType.UNKNOWN
    last_gesture_b: GestureType = GestureType.UNKNOWN

    def as_dict(self) -> dict[str, object]:
        return {
            "volume_a": round(self.volume_a, 3),
            "volume_b": round(self.volume_b, 3),
            "crossfader": round(self.crossfader, 3),
            "playing_a": self.playing_a,
            "playing_b": self.playing_b,
            "bpm": round(self.bpm, 1),
            "tempo_ratio_a": round(self.tempo_ratio_a, 3),
            "tempo_ratio_b": round(self.tempo_ratio_b, 3),
            "nudge_a": round(self.nudge_a, 1),
            "nudge_b": round(self.nudge_b, 1),
            "sync_tempo_requested": self.sync_tempo_requested,
            "beat_match_requested": self.beat_match_requested,
            "sync_status": self.sync_status,
            "transition_phase": self.transition_phase,
            "transition_progress": round(self.transition_progress, 3),
            "transition_beats_remaining": round(self.transition_beats_remaining, 1),
            "transition_label": self.transition_label,
            "transition_status": self.transition_status,
            "filter_master": round(self.filter_master, 3),
            "pinch_state": self.pinch_state,
            "pinch_ratio": round(self.pinch_ratio, 3),
            "effects": sorted(self.effects),
            "last_gesture_a": self.last_gesture_a.value,
            "last_gesture_b": self.last_gesture_b.value,
        }
