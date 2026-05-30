"""Constants for the Facial Visagism Analysis System."""
from __future__ import annotations

# Supported image formats for input
SUPPORTED_FORMATS: frozenset[str] = frozenset({'.jpg', '.jpeg', '.png'})

# Minimum resolution requirements (width, height)
MIN_RESOLUTION: tuple[int, int] = (200, 200)

# Golden ratio value
GOLDEN_RATIO: float = 1.618

# Deviation threshold for golden ratio comparison (10%)
DEVIATION_THRESHOLD: float = 0.10

# Deviation threshold for facial thirds proportion analysis (10%)
THIRDS_DEVIATION_THRESHOLD: float = 0.10

# Default output directory
OUTPUT_DIR: str = "output"

# Environment variable for dlib model path
ENV_DLIB_MODEL_PATH: str = "DLIB_MODEL_PATH"

# dlib shape predictor filename
DEFAULT_MODEL_RELATIVE_PATH: str = "data/shape_predictor_68_face_landmarks.dat"

# Haar cascade parameters
HAAR_SCALE_FACTOR: float = 1.1
HAAR_MIN_NEIGHBORS: int = 5
HAAR_MIN_FACE_SIZE: tuple[int, int] = (100, 100)

# Face bounding box expansion factor (20%)
BBOX_EXPANSION_FACTOR: float = 0.20

# Minimum face area as fraction of image area for warning
SMALL_FACE_AREA_THRESHOLD: float = 0.10

# Landmark visualization
LANDMARK_POINT_RADIUS: int = 3
LANDMARK_LINE_THICKNESS: int = 2

# Legend configuration
LEGEND_BG_COLOR: tuple[int, int, int] = (50, 50, 50)
LEGEND_ALPHA: float = 0.6
LEGEND_FONT_SCALE: float = 0.5
LEGEND_FONT_THICKNESS: int = 1
LEGEND_MARGIN: int = 10
LEGEND_LINE_HEIGHT: int = 20

# Region indices: maps region name -> list of 0-indexed point indices (68-point model)
# dlib 68-point landmark indices (1-indexed in literature, here 0-indexed):
#   jaw: 1-17 -> 0-16
#   left_eyebrow: 18-22 -> 17-21
#   right_eyebrow: 23-27 -> 22-26
#   nose_bridge: 28-31 -> 27-30
#   nose_tip: 32-36 -> 31-35
#   left_eye: 37-42 -> 36-41
#   right_eye: 43-48 -> 42-47
#   outer_mouth: 49-60 -> 48-59
#   inner_mouth: 61-68 -> 60-67
REGION_INDICES: dict[str, list[int]] = {
    "jaw": list(range(0, 17)),
    "left_eyebrow": list(range(17, 22)),
    "right_eyebrow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose_tip": list(range(31, 36)),
    "left_eye": list(range(36, 42)),
    "right_eye": list(range(42, 48)),
    "outer_mouth": list(range(48, 60)),
    "inner_mouth": list(range(60, 68)),
}

# Region connections: maps region name -> list of (i, j) connection pairs
# where i, j are indices WITHIN that region's point list
REGION_CONNECTIONS: dict[str, list[tuple[int, int]]] = {
    "jaw": [(i, i + 1) for i in range(16)],  # open line
    "left_eyebrow": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "right_eyebrow": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "nose_bridge": [(0, 1), (1, 2), (2, 3)],
    "nose_tip": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "left_eye": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],  # closed loop
    "right_eye": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],  # closed loop
    "outer_mouth": [(i, i + 1) for i in range(11)] + [(11, 0)],  # closed loop
    "inner_mouth": [(i, i + 1) for i in range(7)] + [(7, 0)],  # closed loop
}

# Region colors in BGR format
REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "jaw": (0, 255, 0),              # Green
    "left_eyebrow": (255, 255, 0),    # Cyan
    "right_eyebrow": (0, 255, 255),   # Yellow
    "nose_bridge": (255, 0, 255),     # Magenta
    "nose_tip": (255, 0, 255),        # Magenta
    "left_eye": (255, 0, 0),          # Blue
    "right_eye": (0, 165, 255),       # Orange
    "outer_mouth": (0, 0, 255),       # Red
    "inner_mouth": (0, 0, 255),       # Red
}

# Region labels for legend
REGION_LABELS: dict[str, str] = {
    "jaw": "Jaw",
    "left_eyebrow": "Left Brow",
    "right_eyebrow": "Right Brow",
    "nose_bridge": "Nose Bridge",
    "nose_tip": "Nose Tip",
    "left_eye": "Left Eye",
    "right_eye": "Right Eye",
    "outer_mouth": "Outer Mouth",
    "inner_mouth": "Inner Mouth",
}

# Upward expansion of forehead ROI above face bounding box (as fraction of face height)
HAIRLINE_ROI_UPWARD_EXPANSION: float = 0.25

# Canny edge detection parameters for hairline detection
HAIRLINE_CANNY_LOW: int = 30
HAIRLINE_CANNY_HIGH: int = 60
HAIRLINE_GAUSSIAN_KSIZE: int = 5
HAIRLINE_CLOSE_KSIZE: int = 7

# Hairline visualization — dashed line style (BGR color)
HAIRLINE_COLOR: tuple[int, int, int] = (0, 255, 255)  # Yellow
HAIRLINE_DASH_LENGTH: int = 15
HAIRLINE_GAP_LENGTH: int = 8
HAIRLINE_LINE_THICKNESS: int = 2

# Landmark labeler GUI constants
LABELER_WINDOW_NAME: str = "Landmark Labeler"
LABELER_ACTIVE_RADIUS: int = 8
LABELER_PLACED_RADIUS: int = 3
LABELER_PREDICTED_RADIUS: int = 2
LABELER_ACTIVE_COLOR: tuple[int, int, int] = (0, 0, 255)  # Red
LABELER_PREDICTED_COLOR: tuple[int, int, int] = (128, 128, 128)  # Gray
LABELER_TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)  # White
LABELER_BG_COLOR: tuple[int, int, int] = (0, 0, 0)  # Black
LABELER_FONT_SCALE: float = 0.5
LABELER_FONT_THICKNESS: int = 1
LABELER_INDEX_FONT_SCALE: float = 0.4
LABELER_HAIRLINE_DASH_LENGTH: int = 20
LABELER_HAIRLINE_GAP_LENGTH: int = 10
LABELER_HAIRLINE_THICKNESS: int = 2
LABELER_INSTRUCTION_FONT_SCALE: float = 0.5
LABELER_INSTRUCTION_LINE_HEIGHT: int = 20
LABELER_INSTRUCTION_MARGIN: int = 10
LABELER_AUTO_SAVE_DIR: str = "ground_truth"
