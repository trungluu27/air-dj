from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BEATS_PER_BAR = 4
PHRASE_BEATS = 16

# Tracks people actually mix live sit in this range; anything outside it is a
# detector octave error rather than a real tempo.
MIN_BPM = 70.0
MAX_BPM = 180.0
DEFAULT_BPM = 120.0


@dataclass(frozen=True)
class BeatGrid:
    """A constant-interval beat grid, the way a DJ controller models a track.

    Raw detected beats wander by a few milliseconds each, which is enough to make
    a 16-beat transition drift audibly by the end. Fitting a single interval and a
    single phase to the detected beats gives a grid that stays locked for the whole
    transition, and gives us bar lines - the first beat of each bar - which is what
    two tracks actually have to agree on.
    """

    first_beat: float = 0.0
    interval: float = 0.0
    total_samples: int = 0
    downbeat_offset: int = 0
    sample_rate: int = 44_100
    beats_per_bar: int = BEATS_PER_BAR

    @property
    def is_valid(self) -> bool:
        return self.interval > 0.0 and self.total_samples > 0

    @property
    def bpm(self) -> float:
        if not self.is_valid:
            return 0.0
        return 60.0 * self.sample_rate / self.interval

    @property
    def bar_samples(self) -> float:
        return self.interval * self.beats_per_bar

    def beat_at(self, sample: float) -> float:
        """Continuous beat index at sample; fractional between beats."""
        if not self.is_valid:
            return 0.0
        return (sample - self.first_beat) / self.interval

    def sample_at(self, beat: float) -> float:
        return self.first_beat + beat * self.interval

    def beat_phase(self, sample: float) -> float:
        """Where we sit inside the current beat: 0.0 on the beat, rising to 1.0."""
        if not self.is_valid:
            return 0.0
        return float(self.beat_at(sample) % 1.0)

    def next_boundary(self, sample: float, every: int = 1) -> float:
        """Sample index of the next grid line that is a whole multiple of every
        beats counted from the downbeat.

        Always strictly ahead of sample, so a trigger can never fire on a line we
        have already passed.
        """
        if not self.is_valid:
            return float(sample)
        every = max(1, int(every))
        beat = self.beat_at(sample) - self.downbeat_offset
        # The epsilon keeps a trigger landing exactly on a line from being pushed a
        # whole bar into the future by floating point noise.
        target = (math.floor(beat / every + 1e-6) + 1) * every
        return self.sample_at(target + self.downbeat_offset)

    def next_downbeat(self, sample: float) -> float:
        return self.next_boundary(sample, self.beats_per_bar)

    def samples_until(self, sample: float, every: int = 1) -> float:
        return max(0.0, self.next_boundary(sample, every) - float(sample))

    def phrase_start_at_or_after(self, sample: float, beats: int = PHRASE_BEATS) -> float:
        """First phrase line at or after sample, wrapping to the first one if we
        have run past the end of the track.

        A track dropped in mid-phrase sounds wrong however well its beats line up,
        so the incoming deck is always cued here.
        """
        if not self.is_valid:
            return 0.0
        first_phrase = self.sample_at(self.downbeat_offset)
        while first_phrase < 0.0:
            first_phrase += self.interval * beats
        if sample <= first_phrase:
            return float(first_phrase)
        candidate = self.next_boundary(sample - 1e-3, beats)
        if candidate >= self.total_samples:
            return float(first_phrase)
        return float(candidate)

    def positions_normalised(self, limit: int = 2_048) -> np.ndarray:
        """Beat lines as 0..1 fractions of the track, for the waveform display."""
        if not self.is_valid:
            return np.zeros(0, dtype=np.float32)
        start_beat = math.ceil(-self.first_beat / self.interval)
        count = int((self.total_samples - self.sample_at(start_beat)) / self.interval) + 1
        count = max(0, min(limit, count))
        if not count:
            return np.zeros(0, dtype=np.float32)
        beats = np.arange(start_beat, start_beat + count, dtype=np.float64)
        samples = self.first_beat + beats * self.interval
        return (samples / float(self.total_samples)).astype(np.float32)

    @classmethod
    def from_beats(
        cls,
        beat_samples: np.ndarray | list[float],
        total_samples: int,
        sample_rate: int,
        strengths: np.ndarray | None = None,
        bpm_hint: float = 0.0,
        beats_per_bar: int = BEATS_PER_BAR,
    ) -> BeatGrid:
        beats = np.asarray(beat_samples, dtype=np.float64).ravel()
        beats = beats[(beats >= 0.0) & np.isfinite(beats)]
        if total_samples <= 0:
            return cls(sample_rate=sample_rate, beats_per_bar=beats_per_bar)

        interval = cls._fit_interval(beats, sample_rate, bpm_hint)
        if interval <= 0.0:
            return cls(sample_rate=sample_rate, beats_per_bar=beats_per_bar)

        first_beat = cls._fit_phase(beats, interval)
        interval, first_beat = cls._refine(beats, interval, first_beat)
        offset = cls._fit_downbeat(beats, strengths, first_beat, interval, beats_per_bar)
        return cls(
            first_beat=first_beat,
            interval=interval,
            total_samples=int(total_samples),
            downbeat_offset=offset,
            sample_rate=sample_rate,
            beats_per_bar=beats_per_bar,
        )

    @staticmethod
    def _octave_fold(interval: float, sample_rate: int) -> float:
        if interval <= 0.0:
            return 0.0
        # bpm = 60 * sr / interval, so halving the interval doubles the tempo.
        while 60.0 * sample_rate / interval < MIN_BPM:
            interval /= 2.0
        while 60.0 * sample_rate / interval > MAX_BPM:
            interval *= 2.0
        return interval

    @classmethod
    def _fit_interval(cls, beats: np.ndarray, sample_rate: int, bpm_hint: float) -> float:
        if len(beats) >= 2:
            gaps = np.diff(np.sort(beats))
            gaps = gaps[gaps > 0.0]
            if len(gaps):
                base = float(np.median(gaps))
                # A missed beat shows up as a gap of two or three times the real
                # one; dividing each gap by its nearest whole multiple stops those
                # from dragging the estimate upwards.
                multiples = np.maximum(1.0, np.round(gaps / base))
                folded = gaps / multiples
                return cls._octave_fold(float(np.median(folded)), sample_rate)
        bpm = bpm_hint if bpm_hint > 0.0 else DEFAULT_BPM
        return cls._octave_fold(60.0 * sample_rate / bpm, sample_rate)

    @staticmethod
    def _fit_phase(beats: np.ndarray, interval: float) -> float:
        if not len(beats):
            return 0.0
        # Circular mean of where the beats fall inside one interval. A plain mean
        # would be wrong for beats sitting either side of the wrap point.
        angles = 2.0 * math.pi * (beats % interval) / interval
        mean_angle = math.atan2(
            float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles)))
        )
        phase = (mean_angle / (2.0 * math.pi)) * interval
        return float(phase % interval)

    @staticmethod
    def _refine(
        beats: np.ndarray, interval: float, first_beat: float, passes: int = 2
    ) -> tuple[float, float]:
        """Least-squares refinement of the interval and phase.

        The median-of-gaps estimate is typically tens of samples per beat out,
        which compounds into an audible drift across a 16-beat transition. Once
        each detected beat can be assigned an integer index, a straight line fit
        through them removes almost all of that error.
        """
        if len(beats) < 3 or interval <= 0.0:
            return interval, first_beat
        for _ in range(max(1, passes)):
            indices = np.rint((beats - first_beat) / interval)
            if len(np.unique(indices)) < 2:
                break
            slope, intercept = np.polyfit(indices, beats, 1)
            if not (np.isfinite(slope) and np.isfinite(intercept)) or slope <= 0.0:
                break
            interval = float(slope)
            first_beat = float(intercept)
        # Keep the phase inside the first interval so callers see a stable anchor.
        # A fit that lands a hair below zero would otherwise wrap to a full
        # interval and cue every deck one beat late.
        phase = float(first_beat % interval)
        if interval - phase < 1e-3:
            phase = 0.0
        return interval, phase

    @staticmethod
    def _fit_downbeat(
        beats: np.ndarray,
        strengths: np.ndarray | None,
        first_beat: float,
        interval: float,
        beats_per_bar: int,
    ) -> int:
        if strengths is None or not len(beats) or len(strengths) != len(beats):
            return 0
        indices = np.rint((beats - first_beat) / interval).astype(np.int64) % beats_per_bar
        scores = np.full(beats_per_bar, -np.inf, dtype=np.float64)
        for candidate in range(beats_per_bar):
            selected = np.asarray(strengths, dtype=np.float64)[indices == candidate]
            if len(selected):
                scores[candidate] = float(np.mean(selected))
        if not np.isfinite(scores).any():
            return 0
        return int(np.argmax(scores))
