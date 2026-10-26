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
    transition_beats: int = 16
    pinch_close_ratio: float = 0.25
    pinch_open_ratio: float = 0.40
    pinch_tap_max_seconds: float = 0.35
    pinch_rearm_seconds: float = 0.60
    # Higher than smoothing_alpha because the audio side slews the filter knob
    # per block; smoothing slowly in both places makes a sweep lag the hand.
    filter_smoothing_alpha: float = 0.5
    # How long a locked hand may go missing before control is released. Long
    # enough to ride out a dropped frame, short enough that swapping hands
    # deliberately does not feel stuck.
    hand_release_seconds: float = 0.5


WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

MIDDLE_MCP = 9

INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

DEFAULT_SETTINGS = Settings()
