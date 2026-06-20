from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config.settings import DEFAULT_SETTINGS


@dataclass(frozen=True)
class RecordingResult:
    output_path: Path
    frame_count: int
    audio_sample_count: int


class VideoAudioRecorder:
    def __init__(
        self,
        output_dir: Path,
        fps: int = DEFAULT_SETTINGS.recording_fps,
        sample_rate: int = DEFAULT_SETTINGS.sample_rate,
    ) -> None:
        self.output_dir = output_dir
        self.fps = fps
        self.sample_rate = sample_rate
        self.is_recording = False
        self._video_writer = None
        self._session_name = ""
        self._temp_video_path: Path | None = None
        self._audio_path: Path | None = None
        self._final_path: Path | None = None
        self._frame_count = 0
        self._audio_chunks: list[np.ndarray] = []

    def start(self) -> Path:
        if self.is_recording:
            assert self._final_path is not None
            return self._final_path

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._session_name = time.strftime("airdj-%Y%m%d-%H%M%S")
        self._temp_video_path = self.output_dir / f"{self._session_name}.video.mp4"
        self._audio_path = self.output_dir / f"{self._session_name}.wav"
        self._final_path = self.output_dir / f"{self._session_name}.mp4"
        self._frame_count = 0
        self._audio_chunks = []
        self._video_writer = None
        self.is_recording = True
        return self._final_path

    def write_frame(self, frame_rgb: np.ndarray) -> None:
        if not self.is_recording:
            return
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("frame_rgb must be an RGB image")

        if self._video_writer is None:
            self._video_writer = self._open_video_writer(frame_rgb)

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for video recording.") from exc

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self._video_writer.write(frame_bgr)
        self._frame_count += 1

    def write_audio(self, chunk: np.ndarray) -> None:
        if not self.is_recording:
            return
        if chunk.size == 0:
            return
        self._audio_chunks.append(np.asarray(chunk, dtype=np.float32).copy())

    def stop(self) -> RecordingResult:
        if not self.is_recording:
            return RecordingResult(
                output_path=self._final_path or self.output_dir,
                frame_count=self._frame_count,
                audio_sample_count=self._audio_sample_count(),
            )

        self.is_recording = False
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        assert self._temp_video_path is not None
        assert self._audio_path is not None
        assert self._final_path is not None

        audio_sample_count = self._audio_sample_count()
        if audio_sample_count:
            self._write_audio_file(self._audio_path)
            output_path = self._merge_video_audio()
        else:
            output_path = self._temp_video_path

        return RecordingResult(
            output_path=output_path,
            frame_count=self._frame_count,
            audio_sample_count=audio_sample_count,
        )

    def _open_video_writer(self, frame_rgb: np.ndarray):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for video recording.") from exc

        assert self._temp_video_path is not None
        height, width = frame_rgb.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self._temp_video_path),
            fourcc,
            float(self.fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Cannot open video writer")
        return writer

    def _write_audio_file(self, path: Path) -> None:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("SoundFile is required for recording audio.") from exc

        audio = np.concatenate(self._audio_chunks, axis=0)
        sf.write(str(path), np.clip(audio, -1.0, 1.0), self.sample_rate)

    def _merge_video_audio(self) -> Path:
        assert self._temp_video_path is not None
        assert self._audio_path is not None
        assert self._final_path is not None

        try:
            import imageio_ffmpeg
        except ImportError:
            return self._temp_video_path

        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(self._temp_video_path),
            "-i",
            str(self._audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(self._final_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return self._temp_video_path
        return self._final_path

    def _audio_sample_count(self) -> int:
        return int(sum(chunk.shape[0] for chunk in self._audio_chunks))
