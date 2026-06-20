from __future__ import annotations

import numpy as np


def to_portrait_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError("frame must be an HxWxC image")
    if width <= 0 or height <= 0:
        raise ValueError("target width and height must be positive")

    source_height, source_width = frame.shape[:2]
    target_aspect = width / height
    source_aspect = source_width / source_height

    if source_aspect > target_aspect:
        crop_width = max(1, int(source_height * target_aspect))
        start_x = max(0, (source_width - crop_width) // 2)
        cropped = frame[:, start_x : start_x + crop_width]
    else:
        crop_height = max(1, int(source_width / target_aspect))
        start_y = max(0, (source_height - crop_height) // 2)
        cropped = frame[start_y : start_y + crop_height, :]

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for portrait frame resizing.") from exc

    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
