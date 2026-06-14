# AirDJ - Gesture Controlled DJ Mixer

AirDJ is a Gradio demo that lets you control a two-deck DJ mixer with webcam hand gestures.

This v1 focuses on the smallest set of gestures that matter for a strong demo:

- Left hand vertical position controls Deck A volume.
- Right hand vertical position controls Deck B volume.
- Open palm plays the assigned deck.
- Closed fist pauses the assigned deck.
- Two-hand horizontal midpoint controls the crossfader.

Effects, BPM control, scratch, and track switching are intentionally left for later phases.

## Setup

Use Python 3.11+.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Open the Gradio URL shown in the terminal, usually:

```text
http://127.0.0.1:7860
```

## Demo Flow

1. Allow browser webcam access.
2. Click `Start Camera` to show the annotated webcam stream in `AirDJ Live Camera`.
3. Upload a `.wav` or `.mp3` file for Deck A and Deck B.
4. Click `Load Track A` and `Load Track B`.
5. Click `Start Audio`. This opens the audio output stream; it does not start a paused deck by itself.
6. Use gestures in front of the webcam:
   - Left hand up/down: Deck A volume.
   - Right hand up/down: Deck B volume.
   - Open palm: play.
   - Closed fist: pause.
   - Move both hands left/right together: crossfader.

## Project Structure

```text
app.py
audio/
    deck.py
    mixer.py
    player.py
camera/
    hand_tracker.py
config/
    settings.py
core/
    session.py
gestures/
    gesture_detector.py
    gesture_classifier.py
    gesture_mapping.py
ui/
    gradio_app.py
    overlay.py
assets/tracks/
tests/
```

## Notes

- Audio playback runs on the local machine via SoundDevice.
- Webcam frames are captured locally with OpenCV and displayed as an annotated Gradio stream.
- MediaPipe may report mirrored hand labels depending on browser/camera behavior. If Deck A/B feel swapped, switch camera mirroring in the browser or swap hands for the demo.
