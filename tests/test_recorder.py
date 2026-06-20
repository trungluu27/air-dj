import tempfile
import unittest
from pathlib import Path

import numpy as np

from recording.recorder import VideoAudioRecorder


class VideoAudioRecorderTests(unittest.TestCase):
    def test_recorder_collects_frames_and_audio_until_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VideoAudioRecorder(output_dir=Path(tmpdir), fps=5)

            recorder.start()
            recorder.write_frame(np.zeros((1280, 720, 3), dtype=np.uint8))
            recorder.write_audio(np.ones((512, 2), dtype=np.float32))
            result = recorder.stop()

            self.assertFalse(recorder.is_recording)
            self.assertGreater(result.frame_count, 0)
            self.assertGreater(result.audio_sample_count, 0)

    def test_write_calls_are_ignored_when_not_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VideoAudioRecorder(output_dir=Path(tmpdir), fps=5)

            recorder.write_frame(np.zeros((1280, 720, 3), dtype=np.uint8))
            recorder.write_audio(np.ones((512, 2), dtype=np.float32))

            self.assertFalse(recorder.is_recording)


if __name__ == "__main__":
    unittest.main()
