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
python -m unittest tests.test_gesture_mapping   # single file
python -m unittest tests.test_audio_mixer
python -m unittest tests.test_audio_player
```

No pytest config — all tests use `unittest`.

## Architecture

**Runtime state** lives in `ui/gradio_app.py:AirDJRuntime`, a dataclass that owns all components. The Gradio timer (0.2s) drives the main loop.

**Data flow per frame:**

```
cv2.VideoCapture → HandTracker (MediaPipe) → GestureDetector → GestureClassifier
    → GestureMapper → DJSession → Mixer → AudioPlayer (SoundDevice callback)
    → Overlay (OpenCV annotations) → Gradio UI update
```

**Module responsibilities:**

| Module | Key class | Role |
|--------|-----------|------|
| `camera/hand_tracker.py` | `HandTracker` | Wraps MediaPipe Hands; returns `HandResult` with normalized landmark positions |
| `gestures/gesture_detector.py` | `GestureDetector` | Extracts wrist XY (0–1 normalized) and extended finger count from landmarks |
| `gestures/gesture_classifier.py` | `GestureClassifier` | Maps finger count → `OPEN_PALM` (≥4), `FIST` (≤1), or `UNKNOWN` |
| `gestures/gesture_mapping.py` | `GestureMapper` | Maps hand features to session params; applies exponential smoothing (alpha=0.2) and play/pause cooldown (0.8s) |
| `core/session.py` | `DJSession` | Mutable app state: volumes A/B, crossfader, play states, BPM, effects |
| `audio/deck.py` | `Deck` | Loads WAV/MP3, auto-resamples to 44.1kHz, tracks playback position with looping |
| `audio/mixer.py` | `Mixer` | Holds Deck A + Deck B; `get_chunk()` mixes with crossfader gains; thread-safe via `RLock` |
| `audio/player.py` | `AudioPlayer` | SoundDevice `OutputStream` callback; calls `Mixer.get_chunk()` at 512-sample blocks |
| `ui/overlay.py` | `Overlay` | Draws deck state, volumes, crossfader bar, hand info onto OpenCV frames |
| `config/settings.py` | `Settings` | Frozen dataclass of all constants (frame size, FPS, landmark indices, audio params) |

**Gesture → DJ parameter mapping** (in `GestureMapper.apply()`):
- Left hand height → Deck A volume
- Right hand height → Deck B volume
- Midpoint of both hands horizontal → crossfader
- Open palm → play; closed fist → pause (per deck, with cooldown)

**Audio pipeline**: `AudioPlayer` runs on a background thread via SoundDevice. `Mixer` is protected by `RLock` so the Gradio frame-processing thread and audio callback thread don't race.

## Key Design Patterns

- **Frozen dataclasses** for immutable types: `TrackedHand`, `HandResult`, `Settings`
- **Mutable dataclasses** for state: `DJSession`, `Deck`, `Mixer`, `AirDJRuntime`
- All public functions have full type hints
- `assets/tracks/` is gitignored — audio files must be loaded via the UI at runtime
