from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    frame_width: int = 960
    frame_height: int = 540
    target_fps: int = 25
    smoothing_alpha: float = 0.2
    play_pause_cooldown_seconds: float = 0.8
    sample_rate: int = 44_100
    block_size: int = 512
    output_channels: int = 2
    portrait_width: int = 720
    portrait_height: int = 1280
    recording_fps: int = 5


WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

DEFAULT_SETTINGS = Settings()
