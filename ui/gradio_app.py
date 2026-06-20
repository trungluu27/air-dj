from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from audio.mixer import Mixer
from audio.player import AudioPlayer
from camera.hand_tracker import HandTracker
from config.settings import DEFAULT_SETTINGS
from core.session import DJSession
from gestures.gesture_mapping import GestureMapper
from recording.recorder import VideoAudioRecorder
from ui.overlay import Overlay
from ui.portrait import to_portrait_frame

logger = logging.getLogger(__name__)
CAMERA_REFRESH_SECONDS = 0.2


@dataclass
class AirDJRuntime:
    session: DJSession = field(default_factory=DJSession)
    mixer: Mixer = field(default_factory=Mixer)
    mapper: GestureMapper = field(default_factory=GestureMapper)
    player: AudioPlayer | None = None
    tracker: HandTracker | None = None
    overlay: Overlay | None = None
    camera_capture: Any | None = None
    camera_index: int = 0
    recorder: VideoAudioRecorder = field(
        default_factory=lambda: VideoAudioRecorder(Path("assets/recordings"))
    )
    last_recording_path: Path | None = None

    def __post_init__(self) -> None:
        self.player = AudioPlayer(self.mixer)
        self.player.audio_sinks.append(self.recorder.write_audio)

    def start_camera(self) -> str:
        if self.camera_capture is not None:
            return "Camera is already running"
        try:
            self.camera_capture = self._open_camera_capture()
            return "Camera started"
        except Exception as exc:
            logger.exception("Failed to start camera")
            self.camera_capture = None
            return f"Camera failed: {exc}"

    def stop_camera(self) -> str:
        if self.camera_capture is None:
            return "Camera is already stopped"
        self.camera_capture.release()
        self.camera_capture = None
        return "Camera stopped"

    def process_camera_frame(self) -> tuple[Any, dict[str, object], float, float, float, str, str]:
        if self.camera_capture is None:
            return self._frame_response(
                _placeholder_frame("Click Start Camera"),
                {"status": "Camera stopped. Click Start Camera."},
            )

        ok, frame = self.camera_capture.read()
        if not ok or frame is None:
            return self._frame_response(
                _placeholder_frame("No camera frame"),
                {"status": "No camera frame available"},
            )

        try:
            import cv2
        except ImportError as exc:
            return self._frame_response(
                _placeholder_frame("OpenCV missing"),
                {"error": str(exc)},
            )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = to_portrait_frame(
            rgb_frame,
            DEFAULT_SETTINGS.portrait_width,
            DEFAULT_SETTINGS.portrait_height,
        )
        return self.process_frame(rgb_frame)

    def process_frame(self, frame: Any) -> tuple[Any, dict[str, object], float, float, float, str, str]:
        if frame is None:
            return self._frame_response(
                _placeholder_frame("Waiting for frame"),
                {"status": "Waiting for webcam frame"},
            )

        try:
            self._ensure_video_components()
            assert self.tracker is not None
            assert self.overlay is not None
            result = self.tracker.process_rgb(frame)
            features = [hand.features for hand in result.hands]
            self.mapper.apply(self.session, features, now=time.monotonic())
            self.mixer.apply_session(self.session)
            annotated = self.overlay.draw(
                result.annotated_frame,
                self.session,
                features,
                status=f"{len(features)} hand(s)",
            )
            self.recorder.write_frame(annotated)
            return (
                annotated,
                self.session.as_dict(),
                self.session.volume_a,
                self.session.volume_b,
                self.session.crossfader,
                self._deck_state("A"),
                self._deck_state("B"),
            )
        except Exception as exc:  # Gradio callbacks should not crash the app.
            logger.exception("Frame processing failed")
            return self._frame_response(frame, {"error": str(exc)})

    def load_track(self, deck_name: str, file_obj: Any) -> tuple[str, str, str]:
        if file_obj is None:
            return self._message_with_decks(f"Deck {deck_name}: no file selected")
        path = Path(getattr(file_obj, "name", file_obj))
        try:
            self.mixer.load_deck(deck_name, path)
            self.mixer.apply_session(self.session)
            logger.info("track_loaded", extra={"deck": deck_name, "path": str(path)})
            return self._message_with_decks(f"Deck {deck_name}: loaded {path.name}")
        except Exception as exc:
            logger.exception("Failed to load track")
            return self._message_with_decks(
                f"Deck {deck_name}: failed to load {path.name}: {exc}"
            )

    def start_audio(self) -> str:
        try:
            assert self.player is not None
            self.player.start()
            return f"Audio output started. {self._playback_hint()}"
        except Exception as exc:
            logger.exception("Failed to start audio")
            return f"Audio output failed: {exc}"

    def stop_audio(self) -> str:
        try:
            assert self.player is not None
            self.player.stop()
            return "Audio output stopped"
        except Exception as exc:
            logger.exception("Failed to stop audio")
            return f"Audio output stop failed: {exc}"

    def start_recording(self) -> tuple[str, None]:
        if self.camera_capture is None:
            return "Start Camera before recording.", None
        path = self.recorder.start()
        self.last_recording_path = None
        return f"Recording started: {path.name}", None

    def stop_recording(self) -> tuple[str, str | None]:
        result = self.recorder.stop()
        self.last_recording_path = result.output_path
        message = (
            f"Recording saved: {result.output_path.name} "
            f"({result.frame_count} frames, {result.audio_sample_count} audio samples)"
        )
        return message, str(result.output_path)

    def _ensure_video_components(self) -> None:
        if self.tracker is None:
            self.tracker = HandTracker()
        if self.overlay is None:
            self.overlay = Overlay()

    def _open_camera_capture(self) -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for camera capture.") from exc

        capture = cv2.VideoCapture(self.camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_SETTINGS.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_SETTINGS.frame_height)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")
        return capture

    def _frame_response(
        self,
        frame: Any,
        status_update: dict[str, object],
    ) -> tuple[Any, dict[str, object], float, float, float, str, str]:
        return (
            frame,
            self.session.as_dict() | status_update,
            self.session.volume_a,
            self.session.volume_b,
            self.session.crossfader,
            self._deck_state("A"),
            self._deck_state("B"),
        )

    def _deck_state(self, deck_name: str) -> str:
        deck = self.mixer.deck_a if deck_name == "A" else self.mixer.deck_b
        assert deck is not None
        state = "PLAYING" if deck.playing else "PAUSED"
        return f"Deck {deck_name}: {state} - {deck.track_name}"

    def _message_with_decks(self, message: str) -> tuple[str, str, str]:
        return message, self._deck_state("A"), self._deck_state("B")

    def _playback_hint(self) -> str:
        assert self.mixer.deck_a is not None
        assert self.mixer.deck_b is not None

        loaded = [
            deck.name
            for deck in (self.mixer.deck_a, self.mixer.deck_b)
            if deck.has_track
        ]
        playing = [
            deck.name
            for deck in (self.mixer.deck_a, self.mixer.deck_b)
            if deck.playing
        ]

        if not loaded:
            return "No track is loaded yet, so the mixer is silent."
        if not playing:
            return "No deck is playing yet, so the mixer is silent. Show an open palm to play a loaded deck."
        return f"Playing deck(s): {', '.join(playing)}."


def create_demo(runtime: AirDJRuntime | None = None):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required for the AirDJ demo. Install requirements.txt."
        ) from exc

    app = runtime or AirDJRuntime()

    with gr.Blocks(title="AirDJ - Gesture Controlled DJ Mixer") as demo:
        gr.Markdown(
            "# AirDJ - Gesture Controlled DJ Mixer\n"
            "Use webcam gestures: left hand height controls Deck A volume, "
            "right hand height controls Deck B volume, open palm plays, fist pauses, "
            "and two-hand horizontal midpoint controls the crossfader."
        )

        timer = gr.Timer(CAMERA_REFRESH_SECONDS)

        with gr.Row():
            camera_view = gr.Image(
                type="numpy",
                label="AirDJ Live Camera",
                value=_placeholder_frame("Click Start Camera"),
                height=720,
                width=405,
            )

        with gr.Row():
            status = gr.JSON(label="Session Status")
            with gr.Column():
                deck_a_status = gr.Textbox(label="Deck A", interactive=False)
                deck_b_status = gr.Textbox(label="Deck B", interactive=False)
                volume_a = gr.Slider(0, 1, value=app.session.volume_a, label="Volume A", interactive=False)
                volume_b = gr.Slider(0, 1, value=app.session.volume_b, label="Volume B", interactive=False)
                crossfader = gr.Slider(0, 1, value=app.session.crossfader, label="Crossfader", interactive=False)

        with gr.Row():
            track_a = gr.File(label="Upload Track A", file_types=[".wav", ".mp3"])
            track_b = gr.File(label="Upload Track B", file_types=[".wav", ".mp3"])
        with gr.Row():
            load_a = gr.Button("Load Track A")
            load_b = gr.Button("Load Track B")
            start_camera = gr.Button("Start Camera")
            stop_camera = gr.Button("Stop Camera")
            capture_frame = gr.Button("Capture One Frame")
            start_audio = gr.Button("Start Audio")
            stop_audio = gr.Button("Stop Audio")
            start_recording = gr.Button("Start Recording")
            stop_recording = gr.Button("Stop Recording")
        message = gr.Textbox(label="Message", interactive=False)
        recording_download = gr.File(label="Download Recording", interactive=False)

        frame_outputs = [
            camera_view,
            status,
            volume_a,
            volume_b,
            crossfader,
            deck_a_status,
            deck_b_status,
        ]

        timer.tick(
            app.process_camera_frame,
            outputs=frame_outputs,
            queue=False,
            trigger_mode="always_last",
            concurrency_limit=1,
        )
        deck_a_name = gr.State("A")
        deck_b_name = gr.State("B")
        load_a.click(
            app.load_track,
            inputs=[deck_a_name, track_a],
            outputs=[message, deck_a_status, deck_b_status],
        )
        load_b.click(
            app.load_track,
            inputs=[deck_b_name, track_b],
            outputs=[message, deck_a_status, deck_b_status],
        )
        start_camera.click(app.start_camera, outputs=message, queue=False)
        stop_camera.click(app.stop_camera, outputs=message, queue=False)
        capture_frame.click(
            app.process_camera_frame,
            outputs=frame_outputs,
            queue=False,
        )
        start_audio.click(app.start_audio, outputs=message, queue=False)
        stop_audio.click(app.stop_audio, outputs=message, queue=False)
        start_recording.click(
            app.start_recording,
            outputs=[message, recording_download],
            queue=False,
        )
        stop_recording.click(
            app.stop_recording,
            outputs=[message, recording_download],
            queue=False,
        )

    return demo


def _placeholder_frame(message: str = "AirDJ") -> np.ndarray:
    frame = np.zeros(
        (DEFAULT_SETTINGS.portrait_height, DEFAULT_SETTINGS.portrait_width, 3),
        dtype=np.uint8,
    )
    frame[:] = (24, 24, 32)
    try:
        import cv2

        cv2.putText(
            frame,
            message,
            (40, DEFAULT_SETTINGS.portrait_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    except ImportError:
        pass
    return frame
