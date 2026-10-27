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
    # Matches the 0.1s video timer. Frames are placed in the recording by
    # timestamp, so this only decides the grid they land on: below the rate the
    # loop achieves, real frames would be thrown away, above it they are padded
    # out with repeats.
    recording_fps: int = 10
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
    # Filter sweep, measured as travel from where the pinch closed rather than
    # from the middle of the frame. The deadzone absorbs tremor; full travel is
    # about a forearm's movement.
    filter_deadzone: float = 0.05
    filter_travel: float = 0.22


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

# Overlay layout, in pixels of the portrait frame. These live here rather than in
# ui/overlay because the gesture zone is derived from them: the hand is only read
# above the waveform panel, so the drawing and the detection cannot be allowed to
# drift apart.
BOTTOM_PANEL_HEIGHT = 190
WAVEFORM_PANEL_HEIGHT = 250
WAVEFORM_PANEL_GAP = 12
HUD_HEIGHT = BOTTOM_PANEL_HEIGHT + WAVEFORM_PANEL_HEIGHT + WAVEFORM_PANEL_GAP

# Hands are only read above the top edge of the waveform panel. Below it the frame
# is HUD: the hand is behind the graphics, so you cannot see what you are aiming,
# and a gesture there would be guesswork.
GESTURE_ZONE_BOTTOM = 1.0 - HUD_HEIGHT / DEFAULT_SETTINGS.portrait_height
