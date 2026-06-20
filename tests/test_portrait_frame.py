import unittest

import numpy as np

from ui.portrait import to_portrait_frame


class PortraitFrameTests(unittest.TestCase):
    def test_landscape_frame_is_center_cropped_and_resized_to_vertical_9_16(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        portrait = to_portrait_frame(frame, width=720, height=1280)

        self.assertEqual(portrait.shape, (1280, 720, 3))

    def test_center_crop_keeps_center_content(self) -> None:
        frame = np.zeros((4, 8, 3), dtype=np.uint8)
        frame[:, 3:5] = (255, 0, 255)

        portrait = to_portrait_frame(frame, width=2, height=4)

        self.assertTrue(np.any(np.all(portrait == (255, 0, 255), axis=2)))


if __name__ == "__main__":
    unittest.main()
