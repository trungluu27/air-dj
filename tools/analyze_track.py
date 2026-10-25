"""Plan where to cut a long DJ mix into segments AirDJ can mix cleanly.

A commercial DJ mix is the ideal source for AirDJ: it holds one tempo throughout
and its tracks are already in compatible keys, so every segment cut from it plays
back at a tempo ratio of exactly 1.0. The catch is *where* the cuts land. AirDJ has
to work out which beat is the first of the bar separately for every file it loads,
and it only agrees with itself when the files all start on a bar line. Cut at
arbitrary times and the segments disagree by one to three beats, which puts the
kick of one track on the wrong beat of the other.

This tool finds the grid, picks cut points on bar lines, flags the places where the
original DJ was already mid-transition, and can verify its own plan by cutting the
segments in memory and checking that AirDJ reads the same bar line in each.

Usage:
    python tools/analyze_track.py mix.mp3
    python tools/analyze_track.py mix.mp3 --beats 64 --count 6 --verify
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio.beatgrid import BEATS_PER_BAR, BeatGrid  # noqa: E402
from audio.deck import Deck  # noqa: E402

# A kick lives below this; scoring downbeats on a bass-only onset envelope stops
# hi-hats on the offbeats from stealing the vote, which is what makes the app's
# own full-spectrum guess drift by a beat.
KICK_MAX_HZ = 150.0
# How close to one of the original DJ's own transitions is too close to cut.
BOUNDARY_GUARD_BARS = 8


@dataclass(frozen=True)
class Segment:
    index: int
    start_seconds: float
    end_seconds: float
    start_beat: float
    beats: float
    near_boundary: bool

    @property
    def length_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass
class Plan:
    path: Path
    sample_rate: int
    duration_seconds: float
    bpm: float
    grid: BeatGrid
    first_downbeat_seconds: float
    tempo_thirds: list[float]
    boundary_seconds: list[float]
    segments: list[Segment]
    segment_beats: int

    @property
    def tempo_is_stable(self) -> bool:
        if len(self.tempo_thirds) < 2:
            return True
        return (max(self.tempo_thirds) - min(self.tempo_thirds)) < 1.0

    @property
    def clean_segments(self) -> list[Segment]:
        return [segment for segment in self.segments if not segment.near_boundary]


def analyze(path: Path, segment_beats: int = 32) -> Plan:
    import librosa

    audio, sample_rate = librosa.load(str(path), sr=None, mono=False)
    if audio.ndim == 1:
        mono = audio
    else:
        mono = np.mean(audio, axis=0)
    mono = np.ascontiguousarray(mono.astype(np.float32))

    grid, bpm = _fit_grid(mono, sample_rate)
    boundaries = _track_changes(mono, sample_rate, grid)
    segments = _plan_segments(grid, boundaries, segment_beats, sample_rate, len(mono))

    return Plan(
        path=path,
        sample_rate=sample_rate,
        duration_seconds=len(mono) / sample_rate,
        bpm=bpm,
        grid=grid,
        first_downbeat_seconds=grid.sample_at(grid.downbeat_offset) / sample_rate,
        tempo_thirds=_tempo_thirds(mono, sample_rate),
        boundary_seconds=boundaries,
        segments=segments,
        segment_beats=segment_beats,
    )


def _fit_grid(mono: np.ndarray, sample_rate: int) -> tuple[BeatGrid, float]:
    """Run the app's own analysis, so what this tool reports is what AirDJ sees.

    Fitting the grid a second way here would be worse than useless: the cut points
    would be planned against a grid the app does not share.
    """
    deck = Deck("analysis", sample_rate)
    deck.load_array(mono[:, None], sample_rate)
    return deck.beat_grid, deck.bpm


def _tempo_thirds(mono: np.ndarray, sample_rate: int) -> list[float]:
    """Tempo measured over each third of the mix.

    Cutting on a single constant grid assumes one tempo for the whole file. A mix
    that speeds up partway through breaks that assumption, and it is better to say
    so than to hand back cut points that drift.
    """
    import librosa

    third = len(mono) // 3
    if third < sample_rate * 5:
        return []
    tempos: list[float] = []
    for index in range(3):
        chunk = mono[index * third : (index + 1) * third]
        try:
            tempo = librosa.feature.rhythm.tempo(y=chunk, sr=sample_rate)
        except AttributeError:  # librosa < 0.10
            tempo = librosa.beat.tempo(y=chunk, sr=sample_rate)
        tempos.append(round(float(np.asarray(tempo).ravel()[0]), 1))
    return tempos


def _track_changes(mono: np.ndarray, sample_rate: int, grid: BeatGrid) -> list[float]:
    """Find where the original DJ was changing tracks.

    A new track means a new set of instruments, so the timbre shifts. We score on
    MFCCs alone and deliberately ignore harmony: chords change every few bars
    *within* a track, and including chroma made this flag every chord progression
    as a track change while missing the real ones.

    Segments that straddle a boundary contain two songs crossfading, with the
    original fader move already baked in, and mix into mud.
    """
    import librosa

    if not grid.is_valid:
        return []

    hop = 512
    mfcc = librosa.feature.mfcc(y=mono, sr=sample_rate, hop_length=hop, n_mfcc=20)
    features = _normalise(mfcc)

    bar_samples = grid.bar_samples
    bar_count = int((len(mono) - grid.first_beat) / bar_samples)
    if bar_count < 40:
        # Too short for a boundary search to mean anything; the caller just gets
        # no flags rather than a list of noise.
        return []

    edges = [
        int(round((grid.first_beat + index * bar_samples) / hop))
        for index in range(bar_count + 1)
    ]
    edges = [min(max(0, edge), features.shape[1]) for edge in edges]
    per_bar = np.stack(
        [
            features[:, start:end].mean(axis=1)
            if end > start
            else np.zeros(features.shape[0])
            for start, end in zip(edges, edges[1:])
        ]
    )

    # Compare the sixteen bars before each point with the sixteen after. A short
    # window reacts to every breakdown and drop; a track change needs a contrast
    # that holds either side of it, and sixteen bars also brackets the eight-bar
    # crossfade a DJ typically uses.
    window = 16
    novelty = np.zeros(len(per_bar))
    for index in range(window, len(per_bar) - window):
        before = per_bar[index - window : index].mean(axis=0)
        after = per_bar[index : index + window].mean(axis=0)
        novelty[index] = _cosine_distance(before, after)

    if not np.any(novelty):
        return []
    threshold = float(np.mean(novelty) + 1.5 * np.std(novelty))
    peaks = [
        index
        for index in range(window, len(novelty) - window)
        if novelty[index] > threshold
        and novelty[index] == float(np.max(novelty[index - window : index + window]))
    ]
    return [
        float((grid.first_beat + peak * bar_samples) / sample_rate) for peak in peaks
    ]


def _plan_segments(
    grid: BeatGrid,
    boundaries: list[float],
    segment_beats: int,
    sample_rate: int,
    total_samples: int,
) -> list[Segment]:
    if not grid.is_valid:
        return []
    segment_beats = max(BEATS_PER_BAR, (segment_beats // BEATS_PER_BAR) * BEATS_PER_BAR)
    guard_seconds = BOUNDARY_GUARD_BARS * grid.bar_samples / sample_rate

    segments: list[Segment] = []
    beat = float(grid.downbeat_offset)
    index = 0
    while True:
        start = grid.sample_at(beat)
        end = grid.sample_at(beat + segment_beats)
        if end > total_samples:
            break
        start_seconds = start / sample_rate
        end_seconds = end / sample_rate
        near = any(
            start_seconds - guard_seconds < boundary < end_seconds + guard_seconds
            for boundary in boundaries
        )
        segments.append(
            Segment(
                index=index,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                start_beat=beat,
                beats=float(segment_beats),
                near_boundary=near,
            )
        )
        beat += segment_beats
        index += 1
    return segments


def verify(plan: Plan, picks: list[Segment]) -> tuple[float, list[float]]:
    """Cut the chosen segments in memory and check AirDJ agrees on the bar line.

    Returns the worst disagreement in beats, and each segment's reading. Anything
    much above zero means the segments will mix off-beat, however well the tempos
    match.
    """
    import librosa

    audio, sample_rate = librosa.load(str(plan.path), sr=None, mono=False)
    if audio.ndim > 1:
        audio = audio.T
    else:
        audio = audio[:, None]

    readings: list[float] = []
    for segment in picks:
        start = int(segment.start_seconds * sample_rate)
        end = int(segment.end_seconds * sample_rate)
        deck = Deck("A", sample_rate)
        deck.load_array(audio[start:end], sample_rate)
        grid = deck.beat_grid
        if not grid.is_valid:
            readings.append(float("nan"))
            continue
        declared = (grid.first_beat + grid.downbeat_offset * grid.interval) / grid.interval
        readings.append(float(declared % BEATS_PER_BAR))

    finite = [value for value in readings if np.isfinite(value)]
    if len(finite) < 2:
        return 0.0, readings
    spread = max(
        min(abs(a - b), BEATS_PER_BAR - abs(a - b))
        for i, a in enumerate(finite)
        for b in finite[i + 1 :]
    )
    return float(spread), readings


def _normalise(matrix: np.ndarray) -> np.ndarray:
    centred = matrix - matrix.mean(axis=1, keepdims=True)
    scale = np.maximum(1e-9, centred.std(axis=1, keepdims=True))
    return centred / scale


def _cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        return 0.0
    return float(1.0 - np.dot(first, second) / denominator)


def _timestamp(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes):d}:{remainder:06.3f}"


def _pick_spread(segments: list[Segment], count: int) -> list[Segment]:
    """Choose segments spread across the mix, so a set built from them travels
    through the original's energy curve instead of sitting in one spot."""
    if not segments or count <= 0:
        return []
    if count >= len(segments):
        return segments

    first = segments[0].start_seconds
    last = segments[-1].start_seconds
    targets = np.linspace(first, last, count)
    chosen: list[Segment] = []
    for target in targets:
        # Nearest unused segment to this point in the mix, preferring one that is
        # not back-to-back with an existing pick: two adjacent segments come from
        # the same track and sound near enough identical to waste a slot.
        remaining = [segment for segment in segments if segment not in chosen]
        if not remaining:
            break
        spaced = [
            segment
            for segment in remaining
            if all(abs(segment.index - picked.index) > 1 for picked in chosen)
        ]
        pool = spaced or remaining
        chosen.append(min(pool, key=lambda segment: abs(segment.start_seconds - target)))
    return sorted(chosen, key=lambda segment: segment.index)


def report(plan: Plan, count: int, do_verify: bool) -> None:
    bars = plan.segment_beats // BEATS_PER_BAR
    print(f"\n{plan.path.name}")
    print(f"  duration     {_timestamp(plan.duration_seconds)}")
    print(f"  sample rate  {plan.sample_rate} Hz")
    print(f"  tempo        {plan.bpm:.1f} BPM")
    if plan.tempo_thirds:
        thirds = " / ".join(f"{value:.1f}" for value in plan.tempo_thirds)
        verdict = "steady" if plan.tempo_is_stable else "DRIFTS - cuts will not line up"
        print(f"  tempo thirds {thirds} BPM  ({verdict})")
    print(f"  first bar at {_timestamp(plan.first_downbeat_seconds)}")
    print(f"  one beat is  {60.0 / plan.bpm:.4f} s, one bar {4 * 60.0 / plan.bpm:.4f} s")

    if plan.boundary_seconds:
        joined = ", ".join(_timestamp(value) for value in plan.boundary_seconds)
        print(f"\n  likely track changes in the original mix: {joined}")
        print(f"  (segments within {BOUNDARY_GUARD_BARS} bars of one are flagged)")

    print(
        f"\n  {len(plan.segments)} segments of {plan.segment_beats} beats "
        f"({bars} bars, {plan.segments[0].length_seconds:.3f} s each)"
        if plan.segments
        else "\n  no segments could be planned"
    )
    print(f"  {len(plan.clean_segments)} of them clear of a track change\n")
    print("   #   start      end        beats  note")
    for segment in plan.segments:
        note = "near a track change" if segment.near_boundary else ""
        print(
            f"  {segment.index:2d}   {_timestamp(segment.start_seconds):10s} "
            f"{_timestamp(segment.end_seconds):10s} {segment.beats:5.0f}  {note}"
        )

    picks = _pick_spread(plan.clean_segments or plan.segments, count)
    if picks:
        print(f"\n  suggested set of {len(picks)}, spread across the mix:")
        for order, segment in enumerate(picks, start=1):
            print(
                f"    {order}. segment {segment.index:2d}  "
                f"{_timestamp(segment.start_seconds)} -> {_timestamp(segment.end_seconds)}"
            )
        print("\n  ffmpeg:")
        for order, segment in enumerate(picks, start=1):
            print(
                f"    ffmpeg -i \"{plan.path.name}\" -ss {segment.start_seconds:.3f} "
                f"-t {segment.length_seconds:.3f} -c copy seg{order:02d}.mp3"
            )

    if do_verify and picks:
        spread, readings = verify(plan, picks)
        print("\n  verification - what AirDJ reads as the bar line in each cut:")
        for segment, reading in zip(picks, readings):
            print(f"    segment {segment.index:2d}: beat {reading:.3f} of the bar")
        print(f"  worst disagreement: {spread:.3f} beat", end="  ")
        if spread < 0.25:
            print("OK - these will mix on-beat")
        else:
            print("<<< they disagree; try --beats 64, or a different part of the mix")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="the DJ mix to analyse")
    parser.add_argument(
        "--beats",
        type=int,
        default=32,
        help="segment length in beats, rounded to whole bars (default 32)",
    )
    parser.add_argument(
        "--count", type=int, default=5, help="how many segments to suggest (default 5)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="cut the suggested segments in memory and check AirDJ agrees on the bar line",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"no such file: {args.path}", file=sys.stderr)
        return 1

    plan = analyze(args.path, segment_beats=args.beats)
    if not plan.grid.is_valid:
        print("could not find a beat grid in this file", file=sys.stderr)
        return 1
    report(plan, count=args.count, do_verify=args.verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
