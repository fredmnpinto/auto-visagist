"""Type aliases and data classes for the Facial Visagism Analysis System."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

# Type aliases
ImageArray = npt.NDArray[np.uint8]
"""Type alias for an image array (BGR or grayscale uint8)."""

Point = Tuple[int, int]
"""A single (x, y) coordinate pair."""

LandmarksList = List[Point]
"""List of (x, y) landmark coordinates, length 68."""

FaceRect = Tuple[int, int, int, int]
"""Face bounding rectangle: (x, y, width, height)."""

LandmarkRegions = Dict[str, List[Point]]
"""Landmarks grouped by facial region name."""


@dataclass(frozen=True)
class FacialLandmarks:
    """Container for all detected facial landmark data.

    Parameters
    ----------
    image_path : Path
        Path to the input image file.
    face_rect : FaceRect
        Bounding box of the detected face as (x, y, w, h).
    landmarks_68 : LandmarksList
        Full list of 68 landmark (x, y) coordinates in dlib order.
    landmarks_by_region : LandmarkRegions
        Landmarks grouped by facial region.
        Keys: 'jaw', 'left_eyebrow', 'right_eyebrow', 'nose',
              'left_eye', 'right_eye', 'outer_mouth', 'inner_mouth'.
    """

    image_path: Path
    face_rect: FaceRect
    landmarks_68: LandmarksList = field(default_factory=list)
    landmarks_by_region: LandmarkRegions = field(default_factory=dict)
