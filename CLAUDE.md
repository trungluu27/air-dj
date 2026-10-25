# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AirDJ is a gesture-controlled DJ mixer web app. Users control two audio decks (volume, crossfader, play/pause) via real-time hand tracking from a webcam — no keyboard or mouse required.

## Setup & Running

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Launches at `http://127.0.0.1:7860`. Optional flags: `--server-name 0.0.0.0 --server-port 8000 --share`.

## Tests

```bash
python -m unittest discover tests/
python -m unittest tests.test_automix          # single file
python -m unittest tests.test_gesture_mapping
python -m unittest tests.test_pinch
python -m unittest tests.test_filters
python -m unittest tests.test_audio_mixer
python -m unittest tests.test_audio_player
```

No pytest config — all tests use `unittest`.

## Architecture

**Runtime state** lives in `ui/gradio_app.py:AirDJRuntime`, a dataclass that owns all components. The Gradio timer (0.1s) drives the video loop.

**Two clocks.** This is the most important thing to know about the codebase.

*Video clock (0.1s Gradio timer)* — tracking and gesture recognition only:

```
cv2.VideoCapture → HandTracker (MediaPipe) → GestureDetector → PinchDetector
    → GestureMapper → DJSession flags → Mixer.apply_session()
    → Overlay (OpenCV annotations) → Gradio UI update
```

*Audio clock (SoundDevice callback, 512-sample blocks)* — everything musical:

```
AudioPlayer._callback → Mixer.get_chunk() → TransitionEngine.advance()
    → Deck.get_chunk() → bass-cut SweepFilter → equal-power crossfade
    → master SweepFilter → output
```

A transition therefore stays accurate to the sample even though the hand is only sampled ten times a second. Do not move timing decisions into the video loop.

**Module responsibilities:**

| Module | Key class | Role |
|--------|-----------|------|
| `camera/hand_tracker.py` | `HandTracker` | Wraps MediaPipe Hands; returns `HandResult` with normalized landmark positions |
| `gestures/gesture_detector.py` | `GestureDetector` | Extracts wrist XY (0–1 normalized), `pinch_ratio` (thumb-to-index over palm size) and finger states. Takes the frame's `aspect_ratio` — MediaPipe normalizes x and y against different dimensions, so on the 720×1280 portrait frame distances must be corrected before they can be compared |
| `gestures/pinch.py` | `PinchDetector` | Thumb-to-index distance over palm size → tap / hold, with open/close hysteresis. Fires on the **closing edge**, because a quick pinch can fall between two video frames |
| `gestures/gesture_mapping.py` | `GestureMapper` | Pinch tap → transition request; pinch hold + hand height → master filter. Nothing else |
| `gestures/gesture_classifier.py` | `GestureClassifier` | Legacy finger-count classifier, no longer on the control path |
| `core/automix.py` | `TransitionEngine` | Owns the transition: quantise to downbeat, tempo sync (±6%, half/double folding), phrase cue, bass swap, level match, fader curve. Advanced from the audio callback |
| `audio/beatgrid.py` | `BeatGrid` | Constant-interval grid fitted to detected beats by least squares, plus bar and phrase boundaries |
| `audio/filters.py` | `Biquad`, `SweepFilter` | Stateful 24 dB/oct high/low-pass pair via `scipy.signal.lfilter`; state carries across blocks, knob slews per block |
| `core/session.py` | `DJSession` | Mutable app state: volumes A/B, crossfader, play states, BPM, transition state, master filter |
| `audio/deck.py` | `Deck` | Loads WAV/MP3, auto-resamples to 44.1kHz, tracks playback position with looping |
| `audio/mixer.py` | `Mixer` | Holds Deck A + Deck B; `get_chunk()` mixes with crossfader gains; thread-safe via `RLock` |
| `audio/player.py` | `AudioPlayer` | SoundDevice `OutputStream` callback; calls `Mixer.get_chunk()` at 512-sample blocks |
| `ui/overlay.py` | `Overlay` | Draws deck state, volumes, crossfader bar, hand info onto OpenCV frames |
| `config/settings.py` | `Settings` | Frozen dataclass of all constants (frame size, FPS, landmark indices, audio params) |
| `tools/analyze_track.py` | — | Offline prep: plans where to cut a long DJ mix into bar-aligned segments. Calls `Deck.load_array` rather than fitting its own grid, so its plan is based on the grid the app will actually see |

**Downbeat detection** (`Deck._downbeat_scores`) scores candidate bar lines on a
bass-band spectral flux, not the full-spectrum onset envelope. Hi-hats sit on the
offbeats in most dance music and used to outvote the kick, putting the bar line a
whole beat late — invisible on one deck, and enough to make two tracks mix a beat
apart. `tests/test_audio_mixer.py:test_deck_finds_the_downbeat_despite_offbeat_hi_hats`
pins this.

**Gesture mapping** (in `GestureMapper.apply()`), deliberately tiny:
- Pinch closing edge → `session.automix_requested`
- Pinch promoted to a hold → withdraws that request (`automix_cancel_requested`) and sweeps `session.filter_master` from hand height; the centre of the frame is a deadzone
- Hand leaves frame or pinch released → the filter returns to centre

The previous mapping (hand height → volume, two-hand midpoint → crossfader, finger count → play/pause/sync) was removed: it put four meanings on one noisy signal, and an open palm meaning "play" made simply raising a hand start the music.

**Transport ownership.** Once `TransitionEngine.owns_transport` is set, `Mixer.apply_session` reverses direction: deck play states and the crossfader are mirrored *into* the session for display rather than written *from* it. Writing stale video-frame state over the engine would cut a transition off the moment it began.

**Design rule.** The gesture layer only raises requests. Every musical decision — when a mix starts, tempo, cue point, bass swap, fader curve — belongs to `core/automix.py`.

**Audio pipeline**: `AudioPlayer` runs on a background thread via SoundDevice. `Mixer` is protected by `RLock` so the Gradio frame-processing thread and audio callback thread don't race.

## Key Design Patterns

- **Frozen dataclasses** for immutable types: `TrackedHand`, `HandResult`, `Settings`
- **Mutable dataclasses** for state: `DJSession`, `Deck`, `Mixer`, `AirDJRuntime`
- All public functions have full type hints
- `assets/tracks/` is gitignored — audio files must be loaded via the UI at runtime
