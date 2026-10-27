# AirDJ - Gesture Controlled DJ Mixer

AirDJ is a Gradio demo that lets you control a two-deck DJ mixer with webcam hand gestures.

**There is one gesture.** Pinch your thumb and index finger together, and AirDJ
does the DJing:

- **Pinch and release** - mix the two tracks. The first pinch starts Deck A;
  every pinch after that runs a full transition across to the other deck.
- **Pinch and hold, then raise or lower your hand** - sweep a filter. Up thins
  the track out with a high-pass, down muffles it with a low-pass. Let go and it
  returns to centre on its own. The sweep is measured from wherever your hand was
  when you pinched, so holding still does nothing at all, whatever height you
  hold your hand at; only a deliberate move calls off the mix the pinch asked for.

Either hand works, and the first one in frame keeps control - a second hand
drifting into view is ignored until the first one leaves.

That is the whole control surface. You do not need to know how to mix, and you do
not need a second hand in frame.

### What the machine does for you

A transition is not a fader move, it is a sequence, and all of it is automatic:

| Step | What happens |
|------|--------------|
| Quantise | The gesture never fires immediately. It waits for the next bar line, so a mix lands on the beat however sloppily you pinched. |
| Tempo sync | The incoming track is stretched onto the outgoing tempo, capped at ±6% because past that the pitch shift is audible. Half and double time count as matched, so 140 BPM sits happily over 70. |
| Cue | The incoming track starts at a phrase boundary (16 beats), not wherever it happened to be parked. |
| Bar lock | Both beat grids are aligned to the sample, not to the nearest video frame. |
| Bass swap | The incoming track plays with its low end removed until the halfway beat, then the bass hands over. Two kicks and two basslines at once is what makes a mix sound like mud. |
| Level match | The incoming deck is trimmed to the outgoing deck's measured level, up to 12 dB, so the mix does not jump in loudness. |
| Fader | An equal-power crossfade eased in and out over 16 beats, adjustable from 4 to 32 in the UI. |

Progress, the beat countdown and the bass-swap point are all drawn on the camera
view, so you can see a mix coming and see it land.

Each loaded deck also appears in a split beatmatch waveform display with BPM,
tempo, beat grid and playhead. The camera view is rendered in a vertical 9:16
TikTok-style frame with a cyberpunk HUD.

Echo, beat-repeat loops and scratch are intentionally left for later phases.

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
2. Click `Start Camera` to show the annotated 9:16 webcam stream in `AirDJ Live Camera`.
3. Upload a `.wav` or `.mp3` file for Deck A and Deck B.
4. Click `Load Track A` and `Load Track B`.
5. Click `Start Audio`. This opens the audio output stream; it does not start a paused deck by itself.
6. Pinch in front of the webcam:
   - Pinch and release once: Deck A starts playing from a phrase boundary.
   - Pinch and release again: AirDJ mixes Deck A into Deck B over 16 beats.
   - Pinch again: it mixes back the other way.
   - Pinch and hold, then move your hand up or down: sweep the filter. Release to
     return it to centre.
7. Optionally drag `Transition length (beats)` to make mixes shorter or longer.
8. Click `Start Recording` / `Stop Recording` to save a vertical MP4. If audio is playing, the mixed output is merged into the recording.

Beat and BPM analysis uses `librosa`, with a lightweight NumPy fallback for simple
pulse-like tracks. The detected beats are fitted to a constant-interval grid by
least squares, because raw detected beats wander by a few milliseconds each and
that compounds into audible drift across a 16-beat transition.

## Preparing tracks from a DJ mix

A commercial DJ mix is the easiest source to feed AirDJ: it holds one tempo
throughout and its tracks are already in compatible keys, so every segment cut
from it plays at a tempo ratio of exactly 1.0. The catch is that the cuts have to
land on bar lines — AirDJ works out the bar line separately for every file, and it
only agrees with itself when the files all start on one.

```powershell
python tools/analyze_track.py mix.mp3 --beats 64 --count 5 --verify
```

It reports the tempo (and whether it drifts, which would break the whole idea),
finds the first bar line, flags the places where the original DJ was already
mid-transition, suggests a spread of segments, prints the `ffmpeg` commands, and
with `--verify` cuts them in memory to confirm AirDJ reads the same bar line in
each. A worst disagreement near zero means they will mix on-beat.

## Project Structure

```text
app.py
audio/
    beatgrid.py
    deck.py
    filters.py
    mixer.py
    player.py
camera/
    hand_tracker.py
config/
    settings.py
core/
    automix.py
    session.py
gestures/
    gesture_detector.py
    gesture_classifier.py
    gesture_mapping.py
    pinch.py
tools/
    analyze_track.py
ui/
    gradio_app.py
    overlay.py
        portrait.py
    recording/
        recorder.py
assets/tracks/
tests/
```

## Notes

- Audio playback runs on the local machine via SoundDevice.
- Webcam frames are captured locally with OpenCV and displayed as an annotated Gradio stream.
- Recordings are written to `assets/recordings/` and exposed in the `Download Recording` control.
  Frames go into the file by timestamp rather than in arrival order, because the
  video loop does not hit its target rate reliably and the audio stream always
  does: stacking frames up regardless made the picture run at the wrong speed and
  slide further behind the sound the longer the take.
- MediaPipe may report mirrored hand labels depending on browser/camera behavior.
  This no longer matters for control: either hand can pinch, and the right hand
  simply takes priority when both are in frame.
- The transition runs on the audio callback, not on the video timer, so its timing
  is sample-accurate rather than quantised to the frame rate. Pinch detection is
  the only part on the video clock, and the trigger fires on the closing edge so a
  quick pinch cannot fall between two frames.
- Changing tempo resamples, so it shifts pitch. That is why the stretch is capped
  at ±6%; pitch-preserving time stretch would be the next step here.
