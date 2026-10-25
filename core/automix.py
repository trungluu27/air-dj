from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from audio.beatgrid import PHRASE_BEATS
from audio.deck import Deck

logger = logging.getLogger(__name__)

# Past roughly six percent the resampling pitch shift is clearly audible, which
# is why DJs pick tracks whose tempos are already close instead of stretching
# further. We clamp rather than refuse, and say so in the status line.
MAX_TEMPO_STRETCH = 0.06
DEFAULT_TRANSITION_BEATS = 16
# The bass swap happens halfway through, on a beat: two kick drums and two
# basslines playing at once is the single thing that makes a mix sound amateur.
# The handover is kept short - a quarter of a beat - because for its duration both
# decks are partly cut, and a longer overlap audibly drops the level. It is not
# instant only so the filter has somewhere to glide.
BASS_SWAP_RAMP_BEATS = 0.25
# Widest level correction we will apply when matching one track to another. Twelve
# decibels covers the gap between a loud master and a quiet one; measured on real
# tracks, a factor of two was not enough and left the incoming track sounding flat.
MAX_TRIM = 4.0


class TransitionPhase(str, Enum):
    IDLE = "idle"
    ARMED = "armed"
    RUNNING = "running"


@dataclass
class TransitionEngine:
    """Runs a DJ transition on the audio clock.

    A gesture only asks for a transition. Every musical decision - when it starts,
    what tempo the incoming track plays at, where it is cued, when the bass swaps,
    how the fader moves - is made here, so the result lands on the beat no matter
    when the hand actually moved.

    The engine is advanced from inside the audio callback rather than from the UI
    frame loop, so its timeline is sample-accurate instead of being quantised to
    the video frame rate.
    """

    sample_rate: int = 44_100
    beats: int = DEFAULT_TRANSITION_BEATS
    phase: TransitionPhase = TransitionPhase.IDLE
    progress: float = 0.0
    crossfader: float = 0.0
    bass_cut_a: float = 0.0
    bass_cut_b: float = 0.0
    status: str = "ready"
    from_deck: str = ""
    to_deck: str = ""
    beats_remaining: float = 0.0
    trim_a: float = 1.0
    trim_b: float = 1.0
    owns_transport: bool = False
    _outgoing: Deck | None = field(default=None, repr=False)
    _incoming: Deck | None = field(default=None, repr=False)
    _countdown: float = field(default=0.0, repr=False)
    _elapsed: float = field(default=0.0, repr=False)
    _duration: float = field(default=0.0, repr=False)
    _from_value: float = field(default=0.0, repr=False)
    _to_value: float = field(default=1.0, repr=False)

    # ------------------------------------------------------------------ requests

    def request(self, deck_a: Deck, deck_b: Deck, crossfader: float) -> bool:
        """Ask for the next transition. Returns whether the request was taken.

        Called from the UI thread when a pinch is detected; all it does is arm the
        engine, which then waits for a downbeat before anything is audible.
        """
        if self.phase is not TransitionPhase.IDLE:
            self.status = f"transition already running ({self.progress * 100:.0f}%)"
            return False

        outgoing, incoming = self._choose_direction(deck_a, deck_b, crossfader)
        if incoming is None or not incoming.has_track:
            self.status = "load a track on the other deck first"
            return False

        if outgoing is None:
            return self._cold_start(deck_a, deck_b, incoming)

        self._outgoing = outgoing
        self._incoming = incoming
        self.from_deck = outgoing.name
        self.to_deck = incoming.name
        self._from_value = 0.0 if outgoing.name == "A" else 1.0
        self._to_value = 1.0 - self._from_value
        self.crossfader = self._from_value
        self._countdown = self._samples_until_downbeat(outgoing)
        self.phase = TransitionPhase.ARMED
        self.owns_transport = True
        self.progress = 0.0
        self.beats_remaining = float(self.beats)
        self.status = f"cueing deck {incoming.name} on the next downbeat"
        logger.info(
            "transition_armed",
            extra={
                "from": outgoing.name,
                "to": incoming.name,
                "countdown_samples": round(self._countdown),
            },
        )
        return True

    def cancel(self) -> None:
        """Abandon a transition, leaving the mix wherever the fader currently is."""
        if self.phase is TransitionPhase.IDLE:
            return
        self.status = "transition cancelled"
        self._reset_transport()

    # ------------------------------------------------------------------- advance

    def advance(self, frames: int) -> None:
        """Move the timeline on by frames of output audio.

        Called from the audio callback, so it must stay allocation-light and must
        never raise.
        """
        if self.phase is TransitionPhase.IDLE or frames <= 0:
            return

        if self.phase is TransitionPhase.ARMED:
            self._countdown -= frames
            if self._countdown > 0.0:
                return
            # Carry the overshoot into the transition so the timeline does not
            # gain a fraction of a block on every run.
            # The downbeat fell inside the block we are about to render. advance()
            # runs before the decks consume that block, so measured from right now
            # the downbeat is still this far ahead.
            overshoot = -self._countdown
            self._begin(lead_samples=max(0.0, frames - overshoot))
            if self.phase is not TransitionPhase.RUNNING:
                return
            frames = int(min(frames, max(0.0, overshoot)))
            if frames <= 0:
                return

        self._elapsed += frames
        fraction = 1.0 if self._duration <= 0.0 else self._elapsed / self._duration
        fraction = max(0.0, min(1.0, fraction))
        self.progress = fraction
        self.beats_remaining = max(0.0, self.beats * (1.0 - fraction))
        self.crossfader = self._from_value + (self._to_value - self._from_value) * _smoothstep(
            fraction
        )
        self._update_bass_swap(fraction)
        if fraction >= 1.0:
            self._finish()

    def bass_cut_for(self, deck_name: str) -> float:
        return self.bass_cut_a if deck_name == "A" else self.bass_cut_b

    def trim_for(self, deck_name: str) -> float:
        return self.trim_a if deck_name == "A" else self.trim_b

    def _match_gain(self, outgoing: Deck, incoming: Deck) -> None:
        """Bring the incoming track up or down to the outgoing track's level.

        Tracks are mastered at wildly different levels, and a mix that steps in
        loudness halfway through gives the game away more than a slightly late
        beat does. Only the incoming deck is touched, so the master output stays
        where the user left it, and the reference is fixed here at the start
        rather than following the fader across the transition.
        """
        if outgoing.rms <= 1e-6 or incoming.rms <= 1e-6:
            return
        reference = outgoing.rms * self.trim_for(outgoing.name)
        trim = max(1.0 / MAX_TRIM, min(MAX_TRIM, reference / incoming.rms))
        if incoming.name == "A":
            self.trim_a = trim
        else:
            self.trim_b = trim

    # -------------------------------------------------------------------- phases

    def _cold_start(self, deck_a: Deck, deck_b: Deck, incoming: Deck) -> bool:
        """Nothing is playing yet, so the first pinch just drops the needle.

        This is what lets one gesture cover the whole session: first pinch starts
        deck A, every pinch after that mixes across.
        """
        deck_a.pause()
        deck_b.pause()
        self._cue_at_phrase_start(incoming, from_position=0.0)
        incoming.play()
        self.crossfader = 0.0 if incoming.name == "A" else 1.0
        self.bass_cut_a = 0.0
        self.bass_cut_b = 0.0
        self.from_deck = ""
        self.to_deck = incoming.name
        self.progress = 0.0
        self.beats_remaining = 0.0
        self.owns_transport = True
        self.phase = TransitionPhase.IDLE
        self.status = f"playing deck {incoming.name}"
        logger.info("cold_start", extra={"deck": incoming.name})
        return True

    def _begin(self, lead_samples: float = 0.0) -> None:
        outgoing, incoming = self._outgoing, self._incoming
        if outgoing is None or incoming is None:
            self._reset_transport()
            return

        tempo_note = self._sync_tempo(outgoing, incoming)
        self._match_gain(outgoing, incoming)
        self._cue_at_phrase_start(incoming, from_position=incoming.position)
        # The incoming deck is about to render a block that begins before the
        # downbeat, so wind it back to arrive at its cue point exactly on the beat.
        # Without this the two grids start up to a whole block - a good twelve
        # milliseconds - apart, which is enough to hear. The few samples it plays
        # early are at a fader gain of zero.
        if lead_samples > 0.0 and incoming.frame_count:
            tempo = max(0.25, min(2.0, incoming.tempo_ratio))
            incoming.position = (incoming.position - lead_samples * tempo) % incoming.frame_count
        incoming.play()

        # The incoming track arrives with its low end removed; the swap at the
        # halfway point hands the bass over on a beat.
        self.bass_cut_a = 1.0 if incoming.name == "A" else 0.0
        self.bass_cut_b = 1.0 if incoming.name == "B" else 0.0

        self._duration = self._transition_duration(outgoing)
        self._elapsed = 0.0
        self.progress = 0.0
        self.phase = TransitionPhase.RUNNING
        self.status = f"mixing {outgoing.name} into {incoming.name}{tempo_note}"
        logger.info(
            "transition_started",
            extra={
                "from": outgoing.name,
                "to": incoming.name,
                "beats": self.beats,
                "tempo_ratio": round(incoming.tempo_ratio, 4),
            },
        )

    def _finish(self) -> None:
        outgoing, incoming = self._outgoing, self._incoming
        self.crossfader = self._to_value
        if outgoing is not None:
            outgoing.pause()
        self.bass_cut_a = 0.0
        self.bass_cut_b = 0.0
        self.progress = 1.0
        self.beats_remaining = 0.0
        self.phase = TransitionPhase.IDLE
        self._outgoing = None
        self._incoming = None
        if incoming is not None:
            self.status = f"now playing deck {incoming.name}"
        logger.info("transition_finished", extra={"to": self.to_deck})

    def _reset_transport(self) -> None:
        self.phase = TransitionPhase.IDLE
        self.progress = 0.0
        self.beats_remaining = 0.0
        self.bass_cut_a = 0.0
        self.bass_cut_b = 0.0
        self._outgoing = None
        self._incoming = None
        self._elapsed = 0.0
        self._countdown = 0.0

    # -------------------------------------------------------------------- musical

    def _choose_direction(
        self, deck_a: Deck, deck_b: Deck, crossfader: float
    ) -> tuple[Deck | None, Deck | None]:
        if deck_a.playing and not deck_b.playing:
            return deck_a, deck_b
        if deck_b.playing and not deck_a.playing:
            return deck_b, deck_a
        if deck_a.playing and deck_b.playing:
            # Both running: mix away from whichever side the fader already favours.
            return (deck_a, deck_b) if crossfader <= 0.5 else (deck_b, deck_a)
        return None, deck_a if deck_a.has_track else deck_b

    def _sync_tempo(self, outgoing: Deck, incoming: Deck) -> str:
        """Pull the incoming track onto the outgoing track's tempo.

        Half and double time count as matched - a 140 BPM track sits perfectly
        over a 70 BPM one - so we pick whichever octave of the ratio needs the
        least stretching before clamping.
        """
        if not outgoing.bpm or not incoming.bpm:
            return " (no bpm, tempo left alone)"

        target_bpm = outgoing.bpm * outgoing.tempo_ratio
        raw_ratio = target_bpm / incoming.bpm
        ratio = min(
            (raw_ratio * (2.0**octave) for octave in (-2, -1, 0, 1, 2)),
            key=lambda candidate: abs(candidate - 1.0),
        )
        clamped = max(1.0 - MAX_TEMPO_STRETCH, min(1.0 + MAX_TEMPO_STRETCH, ratio))
        incoming.tempo_ratio = clamped
        if abs(clamped - ratio) > 1e-6:
            return f" (tempos {abs(ratio - 1.0) * 100:.0f}% apart, stretch capped)"
        return ""

    def _cue_at_phrase_start(self, deck: Deck, from_position: float) -> None:
        grid = deck.beat_grid
        if not grid.is_valid:
            return
        deck.position = grid.phrase_start_at_or_after(from_position, PHRASE_BEATS)

    def _samples_until_downbeat(self, deck: Deck) -> float:
        """How many output samples until the outgoing track's next bar line.

        Measured in output samples, so a deck playing at a stretched tempo counts
        down at the rate we actually hear.
        """
        grid = deck.beat_grid
        if not grid.is_valid:
            return 0.0
        tempo = max(0.25, min(2.0, deck.tempo_ratio))
        return grid.samples_until(deck.position, grid.beats_per_bar) / tempo

    def _transition_duration(self, outgoing: Deck) -> float:
        grid = outgoing.beat_grid
        tempo = max(0.25, min(2.0, outgoing.tempo_ratio))
        if grid.is_valid:
            return max(1.0, self.beats * grid.interval / tempo)
        # No usable grid: fall back to a plain eight seconds so a transition still
        # completes rather than hanging forever.
        return max(1.0, 8.0 * self.sample_rate)

    def _update_bass_swap(self, fraction: float) -> None:
        if self._incoming is None:
            return
        beat = fraction * self.beats
        swap_beat = self.beats / 2.0
        half_ramp = BASS_SWAP_RAMP_BEATS / 2.0
        handover = _ramp(beat, swap_beat - half_ramp, swap_beat + half_ramp)
        incoming_cut = 1.0 - handover
        outgoing_cut = handover
        if self._incoming.name == "A":
            self.bass_cut_a = incoming_cut
            self.bass_cut_b = outgoing_cut
        else:
            self.bass_cut_b = incoming_cut
            self.bass_cut_a = outgoing_cut


def _smoothstep(fraction: float) -> float:
    """Ease the fader in and out.

    The mixer already applies an equal-power law, so this only softens the moment
    the move starts and the moment it lands.
    """
    fraction = max(0.0, min(1.0, fraction))
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _ramp(value: float, start: float, end: float) -> float:
    if end <= start:
        return 0.0 if value < start else 1.0
    return max(0.0, min(1.0, (value - start) / (end - start)))
