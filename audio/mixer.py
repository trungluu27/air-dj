from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

import numpy as np

from audio.deck import Deck
from audio.filters import SweepFilter, bass_cut_filter
from config.settings import DEFAULT_SETTINGS
from core.automix import TransitionEngine, TransitionPhase
from core.session import DJSession

# Target level the decks are trimmed towards so a transition does not jump in
# loudness when the incoming track is mastered hotter than the outgoing one.
REFERENCE_RMS = 0.12
MAX_TRIM = 2.0


@dataclass
class Mixer:
    sample_rate: int = DEFAULT_SETTINGS.sample_rate
    deck_a: Deck | None = None
    deck_b: Deck | None = None
    crossfader: float = 0.5
    engine: TransitionEngine | None = None
    _lock: RLock = field(default_factory=RLock, repr=False)
    _bass_a: SweepFilter | None = field(default=None, repr=False)
    _bass_b: SweepFilter | None = field(default=None, repr=False)
    _master_filter: SweepFilter | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.deck_a is None:
            self.deck_a = Deck("A", self.sample_rate)
        if self.deck_b is None:
            self.deck_b = Deck("B", self.sample_rate)
        if self.engine is None:
            self.engine = TransitionEngine(sample_rate=self.sample_rate)
        self._bass_a = bass_cut_filter(self.sample_rate)
        self._bass_b = bass_cut_filter(self.sample_rate)
        self._master_filter = SweepFilter(self.sample_rate)

    def apply_session(self, session: DJSession) -> None:
        with self._lock:
            assert self.deck_a is not None
            assert self.deck_b is not None
            assert self.engine is not None
            assert self._master_filter is not None

            if session.automix_cancel_requested:
                self.engine.cancel()
                session.automix_cancel_requested = False
            if session.automix_requested:
                self.engine.beats = max(1, int(session.automix_beats))
                self.engine.request(self.deck_a, self.deck_b, self.crossfader)
                session.automix_requested = False

            self._master_filter.set_target(session.filter_master)

            if session.sync_tempo_requested:
                self._sync_tempo(session)
                session.sync_tempo_requested = False
            if session.beat_match_requested:
                self._align_beat_phase(session)
                session.beat_match_requested = False

            self.deck_a.set_volume(session.volume_a)
            self.deck_b.set_volume(session.volume_b)
            self.deck_a.nudge(session.nudge_a)
            self.deck_b.nudge(session.nudge_b)
            session.nudge_a = 0.0
            session.nudge_b = 0.0

            if self.engine.owns_transport:
                # The engine starts decks, stops them and moves the fader on the
                # audio clock. Writing the previous video frame's session values
                # back over that would cut a transition off the moment it began, so
                # the flow reverses here: decks and fader are mirrored into the
                # session for the UI to display.
                self.crossfader = max(0.0, min(1.0, self.engine.crossfader))
                session.playing_a = self.deck_a.playing
                session.playing_b = self.deck_b.playing
                session.crossfader = self.crossfader
                session.tempo_ratio_a = self.deck_a.tempo_ratio
                session.tempo_ratio_b = self.deck_b.tempo_ratio
            else:
                self.crossfader = max(0.0, min(1.0, session.crossfader))
                self.deck_a.tempo_ratio = session.tempo_ratio_a
                self.deck_b.tempo_ratio = session.tempo_ratio_b
                self.deck_a.playing = session.playing_a and self.deck_a.has_track
                self.deck_b.playing = session.playing_b and self.deck_b.has_track

            self._publish_transition(session)

    def _publish_transition(self, session: DJSession) -> None:
        assert self.engine is not None
        engine = self.engine
        session.transition_phase = engine.phase.value
        session.transition_progress = engine.progress
        session.transition_beats_remaining = engine.beats_remaining
        session.transition_status = engine.status
        session.transition_label = (
            f"{engine.from_deck} to {engine.to_deck}"
            if engine.from_deck and engine.phase is not TransitionPhase.IDLE
            else engine.to_deck
        )

    def get_chunk(self, frame_count: int) -> np.ndarray:
        """Pull the next block of mixed audio.

        This runs on the audio thread, which is also the only clock accurate enough
        to run a transition on, so the engine is advanced from here rather than
        from the video loop.
        """
        with self._lock:
            assert self.deck_a is not None
            assert self.deck_b is not None
            assert self.engine is not None
            assert self._bass_a is not None
            assert self._bass_b is not None
            assert self._master_filter is not None

            self.engine.advance(frame_count)
            if self.engine.owns_transport:
                self.crossfader = max(0.0, min(1.0, self.engine.crossfader))

            left_gain = float(np.cos(self.crossfader * np.pi / 2.0))
            right_gain = float(np.sin(self.crossfader * np.pi / 2.0))

            # The bass swap: whichever deck is arriving has its low end removed
            # until the halfway beat hands it over. Two basslines at once is what
            # turns a mix to mud.
            self._bass_a.set_knob(self.engine.bass_cut_for("A"))
            self._bass_b.set_knob(self.engine.bass_cut_for("B"))
            chunk_a = self._bass_a.process(self.deck_a.get_chunk(frame_count))
            chunk_b = self._bass_b.process(self.deck_b.get_chunk(frame_count))

            mixed = (
                chunk_a * (left_gain * self.engine.trim_for("A"))
                + chunk_b * (right_gain * self.engine.trim_for("B"))
            )
            mixed = self._master_filter.process(mixed)
            return np.clip(mixed, -1.0, 1.0).astype(np.float32)

    def load_deck(self, deck_name: str, path: str | Path) -> None:
        with self._lock:
            self._deck(deck_name).load(path)

    def load_deck_array(
        self, deck_name: str, audio: np.ndarray, sample_rate: int
    ) -> None:
        with self._lock:
            self._deck(deck_name).load_array(audio, sample_rate)

    def _deck(self, deck_name: str) -> Deck:
        assert self.deck_a is not None
        assert self.deck_b is not None
        if deck_name == "A":
            return self.deck_a
        if deck_name == "B":
            return self.deck_b
        raise ValueError(f"Unknown deck: {deck_name}")

    def _master_and_slave(self) -> tuple[str, Deck, Deck]:
        assert self.deck_a is not None
        assert self.deck_b is not None
        if self.crossfader <= 0.5:
            return "A", self.deck_a, self.deck_b
        return "B", self.deck_b, self.deck_a

    def _sync_tempo(self, session: DJSession) -> None:
        master_name, master, slave = self._master_and_slave()
        if not master.bpm or not slave.bpm:
            session.sync_status = "sync unavailable: missing bpm"
            return

        target_bpm = master.effective_bpm
        slave_ratio = max(0.5, min(1.5, target_bpm / slave.bpm))
        if master_name == "A":
            session.tempo_ratio_a = master.tempo_ratio
            session.tempo_ratio_b = slave_ratio
        else:
            session.tempo_ratio_b = master.tempo_ratio
            session.tempo_ratio_a = slave_ratio
        session.sync_status = f"tempo synced to deck {master_name}"

    def _align_beat_phase(self, session: DJSession) -> None:
        master_name, master, slave = self._master_and_slave()
        if not master.has_track or not slave.has_track:
            session.sync_status = "beatmatch unavailable: missing track"
            return
        if len(master.beat_positions) < 2 or len(slave.beat_positions) < 2:
            session.sync_status = "beatmatch unavailable: missing beat grid"
            return

        master_phase = self._beat_phase_fraction(master)
        slave_target = self._position_for_phase(slave, master_phase)
        slave.nudge(slave_target - slave.position)
        session.sync_status = f"beat matched to deck {master_name}"

    def _beat_phase_fraction(self, deck: Deck) -> float:
        beat_samples = np.asarray(deck.beat_positions, dtype=np.float32) * len(deck._audio)
        previous_beats = beat_samples[beat_samples <= deck.position]
        previous = float(previous_beats[-1]) if len(previous_beats) else float(beat_samples[0])
        future_beats = beat_samples[beat_samples > previous]
        next_beat = float(future_beats[0]) if len(future_beats) else previous + self._median_beat_interval(beat_samples)
        interval = max(1.0, next_beat - previous)
        return max(0.0, min(1.0, (deck.position - previous) / interval))

    def _position_for_phase(self, deck: Deck, phase: float) -> float:
        beat_samples = np.asarray(deck.beat_positions, dtype=np.float32) * len(deck._audio)
        previous_beats = beat_samples[beat_samples <= deck.position]
        previous = float(previous_beats[-1]) if len(previous_beats) else float(beat_samples[0])
        future_beats = beat_samples[beat_samples > previous]
        next_beat = float(future_beats[0]) if len(future_beats) else previous + self._median_beat_interval(beat_samples)
        interval = max(1.0, next_beat - previous)
        return float((previous + phase * interval) % len(deck._audio))

    def _median_beat_interval(self, beat_samples: np.ndarray) -> float:
        intervals = np.diff(beat_samples)
        intervals = intervals[intervals > 0.0]
        if not len(intervals):
            return float(self.sample_rate / 2)
        return float(np.median(intervals))
