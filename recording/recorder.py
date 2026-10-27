from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from config.settings import DEFAULT_SETTINGS

# How far the audio sink may fall behind wall clock before the gap is taken to be
# real - the output stream stopped or had not started yet - rather than ordinary
# callback jitter. One audio block is ~12 ms, so this is wide enough never to fire
# on a late callback and narrow enough to catch a stream that was not running.
AUDIO_GAP_TOLERANCE_SECONDS = 0.25


@dataclass(frozen=True)
class RecordingResult:
    output_path: Path
    frame_count: int
    audio_sample_count: int


class VideoAudioRecorder:
    """Records the annotated camera view with the mixed audio.

    The two sides arrive on different clocks, which is the whole difficulty. Audio
    comes from the output stream and so is always exactly real time. Video comes
    from the Gradio timer, which only *aims* at its interval - a frame costs a
    camera read, a MediaPipe pass, an overlay and a JPEG encode, and the rate it
    actually achieves moves around with the machine.

    So frames are placed by timestamp rather than written one after another.
    Writing them in arrival order into a file stamped with a fixed frame rate
    makes the picture play at ``real_rate / fps`` speed, which drifts further out
    of sync with the sound every second; with ``-shortest`` on the merge, the tail
    of the picture is simply discarded. Placing each frame in the slot its
    timestamp belongs to keeps the video stream exactly as long as the wall clock,
    whatever rate the loop managed.
    """

    def __init__(
        self,
        output_dir: Path,
        fps: int = DEFAULT_SETTINGS.recording_fps,
        sample_rate: int = DEFAULT_SETTINGS.sample_rate,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.output_dir = output_dir
        self.fps = fps
        self.sample_rate = sample_rate
        self.clock = clock
        self.is_recording = False
        self._video_writer = None
        self._session_name = ""
        self._temp_video_path: Path | None = None
        self._audio_path: Path | None = None
        self._final_path: Path | None = None
        self._frame_count = 0
        self._audio_chunks: list[np.ndarray] = []
        self._audio_samples = 0
        self._started_at = 0.0
        self._last_frame_bgr: np.ndarray | None = None

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
        self._audio_samples = 0
        self._video_writer = None
        self._last_frame_bgr = None
        self._started_at = self.clock()
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

        slot = self._slot_at(self.clock())
        if slot < self._frame_count:
            # The loop ran faster than the file's frame rate this tick. There is no
            # slot left to put this frame in, so it is dropped rather than pushing
            # every later frame forward in time.
            return

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # Slots skipped while this frame was being produced held the previous
        # picture, so that is what fills them.
        self._fill_to(slot, self._last_frame_bgr if self._frame_count else frame_bgr)
        self._video_writer.write(frame_bgr)
        self._frame_count += 1
        self._last_frame_bgr = frame_bgr

    def write_audio(self, chunk: np.ndarray) -> None:
        if not self.is_recording:
            return
        if chunk.size == 0:
            return

        block = np.asarray(chunk, dtype=np.float32).copy()
        gap = self._audio_gap_at(self.clock(), channels=block.shape[-1])
        if gap is not None:
            self._append_audio(gap)
        self._append_audio(block)

    def stop(self) -> RecordingResult:
        if not self.is_recording:
            return RecordingResult(
                output_path=self._final_path or self.output_dir,
                frame_count=self._frame_count,
                audio_sample_count=self._audio_samples,
            )

        self.is_recording = False
        if self._video_writer is not None:
            # Audio ran until this moment, so the picture has to as well. Without
            # the tail the last fraction of a second - longer if the camera stopped
            # early - would be missing and -shortest would cut the sound to match.
            self._fill_to(self._slot_at(self.clock()), self._last_frame_bgr)
            self._video_writer.release()
            self._video_writer = None

        assert self._temp_video_path is not None
        assert self._audio_path is not None
        assert self._final_path is not None

        audio_sample_count = self._audio_samples
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

    def _slot_at(self, now: float) -> int:
        """Which frame of the file the moment ``now`` belongs to."""
        return int(max(0.0, now - self._started_at) * self.fps)

    def _fill_to(self, slot: int, filler: np.ndarray | None) -> None:
        if filler is None or self._video_writer is None:
            return
        while self._frame_count < slot:
            self._video_writer.write(filler)
            self._frame_count += 1

    def _audio_gap_at(self, now: float, channels: int) -> np.ndarray | None:
        """Silence to put in front of a chunk that arrived late, if any.

        The output stream can start after the recording does, or stop and start
        again during it. Either way the samples that exist no longer line up with
        the clock, and concatenating them regardless would pull the whole soundtrack
        earlier than the picture it belongs with.
        """
        expected = int(max(0.0, now - self._started_at) * self.sample_rate)
        missing = expected - self._audio_samples
        if missing < int(AUDIO_GAP_TOLERANCE_SECONDS * self.sample_rate):
            return None
        return np.zeros((missing, channels), dtype=np.float32)

    def _append_audio(self, block: np.ndarray) -> None:
        self._audio_chunks.append(block)
        self._audio_samples += int(block.shape[0])

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
        return self._audio_samples
