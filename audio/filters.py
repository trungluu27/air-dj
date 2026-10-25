from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

# A fourth-order Butterworth response split into two cascaded biquads. 24 dB/oct
# is what a club mixer's filter knob feels like; a single 12 dB/oct section sounds
# limp by comparison.
BUTTERWORTH_Q: tuple[float, float] = (0.541_196_1, 1.306_562_9)

BYPASS_EPSILON = 1e-4


@dataclass
class Biquad:
    """One second-order IIR section that keeps its state between blocks.

    The audio callback hands us 512-sample blocks. Carrying ``_state`` across
    calls is what stops every block boundary from becoming an audible click.
    """

    sample_rate: int
    channels: int = 2
    _b: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float64), repr=False
    )
    _a: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float64), repr=False
    )
    _state: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 2), dtype=np.float64), repr=False
    )

    def __post_init__(self) -> None:
        self._state = np.zeros((2, self.channels), dtype=np.float64)

    def set_highpass(self, cutoff_hz: float, q: float = 0.707) -> None:
        cos_w0, alpha = self._shape(cutoff_hz, q)
        shared = (1.0 + cos_w0) / 2.0
        self._set_coefficients(
            b=(shared, -2.0 * shared, shared),
            a=(1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha),
        )

    def set_lowpass(self, cutoff_hz: float, q: float = 0.707) -> None:
        cos_w0, alpha = self._shape(cutoff_hz, q)
        shared = (1.0 - cos_w0) / 2.0
        self._set_coefficients(
            b=(shared, 2.0 * shared, shared),
            a=(1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha),
        )

    def process(self, block: np.ndarray) -> np.ndarray:
        if not len(block):
            return block
        filtered, self._state = lfilter(
            self._b, self._a, block.astype(np.float64), axis=0, zi=self._state
        )
        return filtered.astype(np.float32)

    def reset(self) -> None:
        self._state[:] = 0.0

    def _shape(self, cutoff_hz: float, q: float) -> tuple[float, float]:
        nyquist = self.sample_rate / 2.0
        cutoff = max(1.0, min(nyquist * 0.99, float(cutoff_hz)))
        w0 = 2.0 * math.pi * cutoff / self.sample_rate
        return math.cos(w0), math.sin(w0) / (2.0 * max(1e-4, q))

    def _set_coefficients(
        self, b: tuple[float, float, float], a: tuple[float, float, float]
    ) -> None:
        a0 = a[0]
        self._b = np.array([b[0] / a0, b[1] / a0, b[2] / a0], dtype=np.float64)
        self._a = np.array([1.0, a[1] / a0, a[2] / a0], dtype=np.float64)


@dataclass
class SweepFilter:
    """A DJ-style single-knob filter.

    ``knob`` runs -1..1: positive sweeps a high-pass upwards (thins the track out
    and builds tension), negative sweeps a low-pass downwards (muffles it). Both
    branches stay in the signal path at their transparent extremes, so crossing
    zero never produces a coefficient jump.
    """

    sample_rate: int
    channels: int = 2
    min_hp_hz: float = 22.0
    max_hp_hz: float = 4_000.0
    max_lp_hz: float = 18_000.0
    min_lp_hz: float = 170.0
    knob: float = 0.0
    # One exponential step per audio block. The gesture loop only updates the knob
    # a few times a second, so without slewing a sweep would move in audible
    # chunks instead of gliding.
    slew: float = 0.25
    _target_knob: float = field(default=0.0, repr=False)
    _highpass: list[Biquad] = field(default_factory=list, repr=False)
    _lowpass: list[Biquad] = field(default_factory=list, repr=False)
    _tuned_knob: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._highpass = [Biquad(self.sample_rate, self.channels) for _ in BUTTERWORTH_Q]
        self._lowpass = [Biquad(self.sample_rate, self.channels) for _ in BUTTERWORTH_Q]
        self._target_knob = self.knob

    def set_knob(self, knob: float) -> None:
        """Jump straight to a value, for a caller already running on the audio clock."""
        self.knob = max(-1.0, min(1.0, float(knob)))
        self._target_knob = self.knob

    def set_target(self, knob: float) -> None:
        """Glide towards a value, for a caller running on the video clock."""
        self._target_knob = max(-1.0, min(1.0, float(knob)))

    @property
    def target_knob(self) -> float:
        return self._target_knob

    def _advance_knob(self) -> None:
        difference = self._target_knob - self.knob
        if abs(difference) < 1e-4:
            self.knob = self._target_knob
            return
        self.knob += difference * max(0.0, min(1.0, self.slew))

    @property
    def is_bypassed(self) -> bool:
        return abs(self.knob) < BYPASS_EPSILON

    @property
    def highpass_hz(self) -> float:
        amount = max(0.0, self.knob)
        return self._sweep(self.min_hp_hz, self.max_hp_hz, amount)

    @property
    def lowpass_hz(self) -> float:
        amount = max(0.0, -self.knob)
        return self._sweep(self.max_lp_hz, self.min_lp_hz, amount)

    def process(self, block: np.ndarray) -> np.ndarray:
        self._advance_knob()
        if self.is_bypassed:
            # Nothing to do, but the sections must not hold stale history for the
            # next time the knob moves off centre.
            self.reset()
            return block
        self._retune()
        output = block
        for section in self._highpass:
            output = section.process(output)
        for section in self._lowpass:
            output = section.process(output)
        return output

    def reset(self) -> None:
        for section in (*self._highpass, *self._lowpass):
            section.reset()
        self._tuned_knob = None

    def _retune(self) -> None:
        if self._tuned_knob is not None and abs(self._tuned_knob - self.knob) < 1e-4:
            return
        highpass_hz = self.highpass_hz
        lowpass_hz = self.lowpass_hz
        for section, q in zip(self._highpass, BUTTERWORTH_Q):
            section.set_highpass(highpass_hz, q)
        for section, q in zip(self._lowpass, BUTTERWORTH_Q):
            section.set_lowpass(lowpass_hz, q)
        self._tuned_knob = self.knob

    def _sweep(self, start_hz: float, end_hz: float, amount: float) -> float:
        # Geometric interpolation: a filter sweep has to move in octaves to feel
        # linear to the ear.
        amount = max(0.0, min(1.0, amount))
        return float(start_hz * (end_hz / start_hz) ** amount)


def bass_cut_filter(sample_rate: int, channels: int = 2) -> SweepFilter:
    """The filter used for a transition's bass swap.

    Its high-pass tops out around 220 Hz: enough to pull the kick and bassline
    out of one track so the two do not fight, without thinning the track audibly.
    """
    return SweepFilter(
        sample_rate=sample_rate,
        channels=channels,
        min_hp_hz=20.0,
        max_hp_hz=220.0,
        max_lp_hz=20_000.0,
        min_lp_hz=20_000.0,
    )
