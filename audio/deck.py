from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from audio.beatgrid import BeatGrid

# Analysis frame hop, shared by the onset envelope and the bass-band flux so their
# frame indices line up.
HOP_LENGTH = 512
# A kick drum lives below this. Used to score which beat starts a bar.
KICK_MAX_HZ = 150.0


@dataclass
class Deck:
    name: str
    sample_rate: int
    volume: float = 1.0
    playing: bool = False
    position: float = 0.0
    track_name: str = "No track"
    bpm: float = 0.0
    tempo_ratio: float = 1.0
    rms: float = 0.0
    beat_grid: BeatGrid = field(default_factory=BeatGrid, repr=False)
    beat_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32),
        repr=False,
    )
    waveform_peaks: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32),
        repr=False,
    )
    _audio: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float32),
        repr=False,
    )

    def load(self, path: str | Path) -> None:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "SoundFile is required for loading audio. Install requirements.txt."
            ) from exc

        audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
        self.load_array(audio, sample_rate=source_rate)
        self.track_name = Path(path).name

    def load_array(self, audio: np.ndarray, sample_rate: int) -> None:
        normalized = np.asarray(audio, dtype=np.float32)
        if normalized.ndim == 1:
            normalized = normalized[:, None]
        if normalized.shape[1] == 1:
            normalized = np.repeat(normalized, 2, axis=1)
        elif normalized.shape[1] > 2:
            normalized = normalized[:, :2]

        if sample_rate != self.sample_rate and normalized.size:
            normalized = self._resample(normalized, sample_rate, self.sample_rate)

        self._audio = np.clip(normalized, -1.0, 1.0).astype(np.float32)
        self.position = 0.0
        self.tempo_ratio = 1.0
        self.waveform_peaks = self._build_waveform_peaks(self._audio)
        self.rms = self._measure_rms(self._audio)
        self.beat_grid = self._build_beat_grid(self._audio)
        self.bpm = round(self.beat_grid.bpm, 1) if self.beat_grid.is_valid else 0.0
        self.beat_positions = self.beat_grid.positions_normalised()

    def play(self) -> None:
        if self.has_track:
            self.playing = True

    def pause(self) -> None:
        self.playing = False

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, volume))

    def nudge(self, sample_offset: float) -> None:
        if not self.has_track:
            return
        self.position = float((self.position + sample_offset) % len(self._audio))

    def get_chunk(self, frame_count: int) -> np.ndarray:
        if not self.playing or not self.has_track:
            return np.zeros((frame_count, 2), dtype=np.float32)

        tempo = max(0.25, min(2.0, self.tempo_ratio))
        sample_positions = (np.arange(frame_count, dtype=np.float32) * tempo + self.position) % len(self._audio)
        lower_indices = np.floor(sample_positions).astype(np.int64)
        upper_indices = (lower_indices + 1) % len(self._audio)
        weights = (sample_positions - lower_indices)[:, None]
        chunk = self._audio[lower_indices] * (1.0 - weights) + self._audio[upper_indices] * weights
        self.position = float((self.position + frame_count * tempo) % len(self._audio))
        return chunk.astype(np.float32) * self.volume

    @property
    def has_track(self) -> bool:
        return len(self._audio) > 0

    @property
    def frame_count(self) -> int:
        return len(self._audio)

    @property
    def progress(self) -> float:
        if not self.has_track:
            return 0.0
        return float(self.position / len(self._audio))

    @property
    def effective_bpm(self) -> float:
        return self.bpm * max(0.25, min(2.0, self.tempo_ratio)) if self.bpm else 0.0

    def _resample(
        self, audio: np.ndarray, source_rate: int, target_rate: int
    ) -> np.ndarray:
        if source_rate <= 0:
            raise ValueError("source sample rate must be positive")
        duration = len(audio) / source_rate
        target_length = max(1, int(duration * target_rate))
        source_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        target_x = np.linspace(0.0, duration, num=target_length, endpoint=False)
        channels = [
            np.interp(target_x, source_x, audio[:, channel])
            for channel in range(audio.shape[1])
        ]
        return np.stack(channels, axis=1).astype(np.float32)

    def _build_waveform_peaks(self, audio: np.ndarray, bucket_count: int = 256) -> np.ndarray:
        if not len(audio):
            return np.zeros(0, dtype=np.float32)

        mono = np.max(np.abs(audio), axis=1)
        buckets = np.array_split(mono, min(bucket_count, len(mono)))
        peaks = np.array([float(np.max(bucket)) for bucket in buckets], dtype=np.float32)
        max_peak = float(np.max(peaks)) if len(peaks) else 0.0
        if max_peak > 0.0:
            peaks = peaks / max_peak
        return peaks.astype(np.float32)

    def _estimate_bpm(self, audio: np.ndarray) -> float:
        onsets = self._detect_onsets(audio)
        if len(onsets) < 2:
            return 0.0

        intervals = np.diff(onsets) / float(self.sample_rate)
        intervals = intervals[intervals > 0.05]
        if not len(intervals):
            return 0.0

        bpm = 60.0 / float(np.median(intervals))
        while bpm < 80.0:
            bpm *= 2.0
        while bpm > 180.0:
            bpm /= 2.0
        return round(bpm, 1)

    def _build_beat_grid(self, audio: np.ndarray) -> BeatGrid:
        """Fit a beat grid, trusting librosa ahead of the amplitude heuristic.

        The heuristic in ``_detect_onsets`` only picks samples above 60% of the
        peak. On a mastered track that is thousands of scattered points rather
        than beats, so preferring it over a real beat tracker left the grid - and
        anything built on it, like a beat-matched transition - meaningless.
        """
        beats, strengths, bpm_hint = self._analyze_beats_with_librosa(audio)
        if len(beats) < 2:
            beats = self._detect_onsets(audio)
            strengths = None
            bpm_hint = self._estimate_bpm(audio)
        return BeatGrid.from_beats(
            beats,
            total_samples=len(audio),
            sample_rate=self.sample_rate,
            strengths=strengths,
            bpm_hint=bpm_hint,
        )

    def _analyze_beats_with_librosa(
        self, audio: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, float]:
        """Return beat positions in samples, their onset strengths, and the tempo.

        The strengths are what let the grid work out which beat is the ``one`` of
        each bar, since a downbeat is normally the loudest beat in its bar.
        """
        empty = (np.zeros(0, dtype=np.float64), None, 0.0)
        if len(audio) < self.sample_rate:
            return empty

        try:
            import librosa
        except ImportError:
            return empty

        mono = np.mean(audio, axis=1).astype(np.float32)
        if not np.any(mono):
            return empty

        try:
            onset_envelope = librosa.onset.onset_strength(y=mono, sr=self.sample_rate)
            tempo, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_envelope,
                sr=self.sample_rate,
                units="frames",
            )
            beat_samples = librosa.frames_to_samples(beat_frames)
        except Exception:
            return empty

        beat_samples = np.asarray(beat_samples, dtype=np.float64).ravel()
        if not len(beat_samples):
            return empty

        beat_frames = np.asarray(beat_frames, dtype=np.int64).ravel()
        strengths = self._downbeat_scores(mono, beat_frames, len(beat_samples), librosa)

        tempo_value = (
            float(tempo.ravel()[0])
            if isinstance(tempo, np.ndarray) and tempo.size
            else float(tempo) if not isinstance(tempo, np.ndarray) else 0.0
        )
        return beat_samples, strengths, tempo_value

    def _downbeat_scores(
        self,
        mono: np.ndarray,
        beat_frames: np.ndarray,
        beat_count: int,
        librosa: Any,
    ) -> np.ndarray | None:
        """How strongly each beat looks like the first beat of a bar.

        Scored on the bass band alone, because that is where the downbeat actually
        announces itself - on the kick. Scoring on the full spectrum lets hi-hats
        outvote it, and in most dance music the hats sit on the offbeats, which
        pushed the detected bar line a whole beat late. That error is invisible
        while a track is played on its own and ruins a mix the moment a second
        track has to agree with it.
        """
        try:
            spectrum = np.abs(librosa.stft(mono, hop_length=HOP_LENGTH))
            frequencies = librosa.fft_frequencies(sr=self.sample_rate)
        except Exception:
            return None

        band = frequencies <= KICK_MAX_HZ
        if not band.any() or not spectrum.size:
            return None

        energy = spectrum[band].sum(axis=0)
        flux = np.maximum(0.0, np.diff(energy, prepend=energy[:1]))
        in_range = beat_frames[beat_frames < len(flux)]
        if len(in_range) != beat_count:
            return None
        return flux[in_range].astype(np.float64)

    def _measure_rms(self, audio: np.ndarray) -> float:
        """Average level of the track, used to trim one deck against the other so
        a transition does not jump in loudness halfway through."""
        if not len(audio):
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    def _detect_onsets(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < self.sample_rate:
            return np.zeros(0, dtype=np.float32)

        mono = np.max(np.abs(audio), axis=1)
        max_amplitude = float(np.max(mono))
        if max_amplitude <= 0.0:
            return np.zeros(0, dtype=np.float32)

        threshold = max(max_amplitude * 0.6, float(np.mean(mono) + np.std(mono) * 3.0))
        pulse_indices = np.flatnonzero(mono >= threshold)
        if len(pulse_indices) < 2:
            return np.zeros(0, dtype=np.float32)

        groups = np.split(pulse_indices, np.where(np.diff(pulse_indices) > 1)[0] + 1)
        return np.array([group[0] for group in groups if len(group)], dtype=np.float32)
