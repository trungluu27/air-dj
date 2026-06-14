# AIR DJ - Gesture Controlled DJ Mixer

You are a senior Python engineer, computer vision engineer, audio DSP engineer, and UI designer.

Build a complete desktop application called "AirDJ" that allows users to control a virtual DJ mixer using hand gestures captured from a webcam.

## Goal

Create a real-time gesture-controlled DJ mixer using:

* Python 3.11+
* MediaPipe Hands
* OpenCV
* NumPy
* SoundDevice
* SoundFile
* SciPy

The user should be able to load music tracks and control them entirely with hand gestures without touching the keyboard or mouse.

The application must look like a professional DJ console and be suitable for demonstrating AI and Computer Vision capabilities in a portfolio.

---

# Architecture

Use clean architecture and separate modules:

air_dj/

```
app.py

camera/
    hand_tracker.py

gestures/
    gesture_detector.py
    gesture_classifier.py
    gesture_mapping.py

audio/
    deck.py
    mixer.py
    effects.py
    player.py

ui/
    dashboard.py
    widgets.py

assets/
    tracks/

config/
    settings.py
```

Use OOP.

Use type hints everywhere.

Use dataclasses when appropriate.

Avoid global variables.

---

# Core Features

## Webcam Tracking

Capture webcam frames using OpenCV.

Use MediaPipe Hands.

Support:

* One hand
* Two hands

Track:

* wrist
* thumb tip
* index tip
* middle tip
* ring tip
* pinky tip

Display hand landmarks on screen.

Run at minimum 25 FPS.

---

# DJ Mixer

Implement two virtual decks.

Deck A
Deck B

Each deck can load an audio file.

Supported formats:

* wav
* mp3

Both decks can play simultaneously.

---

# Gesture Controls

## Left Hand

Controls Deck A.

### Volume

Vertical hand position controls volume.

Hand at top:

volume = 100%

Hand at bottom:

volume = 0%

Use smoothing to avoid jitter.

---

## Right Hand

Controls Deck B.

### Volume

Vertical hand position controls volume.

Hand at top:

volume = 100%

Hand at bottom:

volume = 0%

Use smoothing.

---

# Crossfader

Use horizontal distance between hands.

When hands move left:

Deck A louder.

When hands move right:

Deck B louder.

Crossfade smoothly.

Display crossfader position.

---

# Play Pause

Open palm:

PLAY

Closed fist:

PAUSE

Detect robustly.

Avoid accidental triggering.

Use gesture cooldown.

---

# Track Switching

One finger:

Track 1

Two fingers:

Track 2

Three fingers:

Track 3

Allow assigning tracks independently to each deck.

---

# Scratch Effect

Create virtual turntables.

When index finger enters deck area:

enable scratching.

Horizontal movement:

simulate vinyl scratching.

Requirements:

* real-time response
* smooth audio feedback
* adjustable sensitivity

---

# Audio Effects

Implement:

1. Echo
2. Reverb
3. Low Pass Filter
4. High Pass Filter
5. Bass Boost

Use SciPy DSP.

Effects should be applied in real time.

Effects can be enabled using gestures.

---

# Gesture Mapping

Open Palm:
Play

Closed Fist:
Pause

Peace Sign:
Toggle Reverb

Rock Sign:
Toggle Echo

Thumb Up:
Bass Boost On

Thumb Down:
Bass Boost Off

---

# BPM Control

Distance between both hands controls playback speed.

Hands close:
100 BPM

Hands medium:
120 BPM

Hands far:
140 BPM

Use interpolation.

Preserve pitch if possible.

If pitch preservation is difficult, implement speed change first.

---

# User Interface

Build a futuristic DJ interface using OpenCV.

Show:

* webcam feed
* hand landmarks
* deck A
* deck B
* crossfader
* volume meters
* active effects
* BPM
* current track

Layout example:

+------------------------------------------------+
| CAMERA VIEW                                    |
|                                                |
|                                                |
+------------------------------------------------+
| DECK A            CROSSFADER          DECK B   |
|                                                |
| volume           BPM               volume      |
|                                                |
| effects                             effects    |
+------------------------------------------------+

UI should update in real time.

---

# Performance

Target:

* 25+ FPS webcam
* low audio latency
* smooth UI

Use threading where needed.

Separate:

* video processing
* gesture processing
* audio engine

---

# Logging

Add structured logging.

Log:

* gesture detected
* track loaded
* effect enabled
* effect disabled

---

# Error Handling

Handle:

* missing webcam
* invalid audio file
* unsupported format
* MediaPipe failure

Application should never crash.

---

# Future Extensions

Design architecture to support:

* YOLO hand detection
* ML gesture classification
* MIDI controller support
* Spotify integration
* OBS integration
* Multi-user mode

---

# Deliverables

Generate:

1. Full project structure
2. All Python source files
3. Requirements.txt
4. README.md
5. Setup instructions
6. Run instructions
7. Comments explaining important sections

Build the project incrementally.

Start with Phase 1:

* Webcam
* MediaPipe
* Hand tracking
* UI

After Phase 1 is complete, continue with Phase 2 automatically.
