import tempfile
import unittest
from pathlib import Path

import numpy as np

from recording.recorder import VideoAudioRecorder

SR = 44_100
BLOCK = 512


class FakeClock:
    """A clock the test drives, so frame timing is exact rather than timing-dependent."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def frame(value: int = 0) -> np.ndarray:
    return np.full((64, 36, 3), value, dtype=np.uint8)


class VideoAudioRecorderTests(unittest.TestCase):
    def test_recorder_collects_frames_and_audio_until_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VideoAudioRecorder(output_dir=Path(tmpdir), fps=5)

            recorder.start()
            recorder.write_frame(frame())
            recorder.write_audio(np.ones((BLOCK, 2), dtype=np.float32))
            result = recorder.stop()

            self.assertFalse(recorder.is_recording)
            self.assertGreater(result.frame_count, 0)
            self.assertGreater(result.audio_sample_count, 0)

    def test_write_calls_are_ignored_when_not_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VideoAudioRecorder(output_dir=Path(tmpdir), fps=5)

            recorder.write_frame(frame())
            recorder.write_audio(np.ones((BLOCK, 2), dtype=np.float32))

            self.assertFalse(recorder.is_recording)


class RecordingStaysInSyncTests(unittest.TestCase):
    """The picture has to last exactly as long as the sound.

    Audio is fed from the output stream, so it is always real time. Video is fed
    from the Gradio timer at whatever rate the machine manages. Writing those
    frames back to back into a file stamped with a fixed rate made the picture run
    at ``real_rate / fps`` speed - drifting further behind the sound every second,
    with the tail cut off by ``-shortest``.
    """

    def setUp(self) -> None:
        self.clock = FakeClock()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.fps = 10
        self.recorder = VideoAudioRecorder(
            output_dir=Path(self.tmpdir.name), fps=self.fps, clock=self.clock
        )

    def record(self, seconds: float, loop_fps: float, with_audio: bool = True) -> None:
        """Run the camera loop and the audio stream side by side for a while."""
        frame_interval = 1.0 / loop_fps
        audio_interval = BLOCK / SR
        next_frame = 0.0
        next_audio = 0.0
        step = 1.0 / SR * 64
        elapsed = 0.0
        index = 0
        while elapsed < seconds:
            if elapsed >= next_frame:
                self.recorder.write_frame(frame(index % 256))
                index += 1
                next_frame += frame_interval
            if with_audio and elapsed >= next_audio:
                self.recorder.write_audio(np.zeros((BLOCK, 2), dtype=np.float32))
                next_audio += audio_interval
            self.clock.advance(step)
            elapsed += step

    def assert_in_sync(self, seconds: float, tolerance: float = 0.15) -> None:
        result = self.recorder.stop()
        video_seconds = result.frame_count / self.fps
        audio_seconds = result.audio_sample_count / SR

        self.assertAlmostEqual(video_seconds, seconds, delta=tolerance)
        self.assertAlmostEqual(audio_seconds, seconds, delta=tolerance)
        self.assertAlmostEqual(
            video_seconds,
            audio_seconds,
            delta=tolerance,
            msg=f"picture {video_seconds:.2f}s against sound {audio_seconds:.2f}s",
        )

    def test_a_slow_video_loop_does_not_stretch_the_picture(self) -> None:
        """Six frames a second must still give one second of video per second.

        Written in arrival order this produced 0.6s of video per second of sound.
        """
        self.recorder.start()
        self.record(seconds=4.0, loop_fps=6.0)

        self.assert_in_sync(4.0)

    def test_a_fast_video_loop_does_not_compress_the_picture(self) -> None:
        self.recorder.start()
        self.record(seconds=4.0, loop_fps=18.0)

        self.assert_in_sync(4.0)

    def test_the_tail_is_held_when_the_camera_stops_early(self) -> None:
        """Audio keeps coming after the last frame, so the picture is held.

        Otherwise the sound outruns the picture and -shortest throws the end away.
        """
        self.recorder.start()
        self.record(seconds=2.0, loop_fps=10.0)
        self.record(seconds=2.0, loop_fps=10.0, with_audio=True)
        # ...and now the camera dies while the music plays on.
        for _ in range(int(2.0 * SR / BLOCK)):
            self.recorder.write_audio(np.zeros((BLOCK, 2), dtype=np.float32))
            self.clock.advance(BLOCK / SR)

        self.assert_in_sync(6.0)

    def test_audio_starting_late_is_padded_rather_than_pulled_forward(self) -> None:
        """Nothing is playing for the first two seconds, then the stream opens.

        Concatenating from the first chunk put the music two seconds ahead of the
        hand that started it.
        """
        self.recorder.start()
        self.record(seconds=2.0, loop_fps=10.0, with_audio=False)
        self.record(seconds=2.0, loop_fps=10.0, with_audio=True)

        result = self.recorder.stop()

        self.assertAlmostEqual(result.audio_sample_count / SR, 4.0, delta=0.15)
        self.assertAlmostEqual(result.frame_count / self.fps, 4.0, delta=0.15)

    def test_ordinary_callback_jitter_does_not_insert_silence(self) -> None:
        """Audio blocks arrive in bursts; only a real gap may be padded."""
        self.recorder.start()
        for index in range(int(3.0 * SR / BLOCK)):
            self.recorder.write_audio(np.ones((BLOCK, 2), dtype=np.float32))
            # Half the blocks arrive late, half early, as they do in bursts.
            self.clock.advance(BLOCK / SR * (1.4 if index % 2 else 0.6))

        result = self.recorder.stop()
        audio = np.concatenate(self.recorder._audio_chunks, axis=0)

        self.assertEqual(result.audio_sample_count, int(3.0 * SR / BLOCK) * BLOCK)
        self.assertTrue(bool(np.all(audio == 1.0)))


if __name__ == "__main__":
    unittest.main()
