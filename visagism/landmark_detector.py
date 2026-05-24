"""Facial landmark detection using dlib 68-point shape predictor."""

from __future__ import annotations

import math
from pathlib import Path

import dlib

from visagism.constants import REGION_INDICES
from visagism.types import (
    FaceRect,
    FacialLandmarks,
    ImageArray,
    LandmarkRegions,
    LandmarksList,
)


class LandmarkDetector:
    """Detects 68 facial landmarks using the dlib shape predictor.

    Wraps the dlib ``shape_predictor`` and returns a ``FacialLandmarks``
    dataclass containing the full 68-point list and points grouped by
    facial region.
    """

    def __init__(self, model_path: Path) -> None:
        """Initialise the landmark detector.

        Parameters
        ----------
        model_path : Path
            Path to the dlib shape predictor model file
            (``shape_predictor_68_face_landmarks.dat``).
        """
        self._predictor = dlib.shape_predictor(str(model_path))

    def detect(
        self,
        img_gray: ImageArray,
        face_rect: FaceRect,
        image_path: Path,
    ) -> FacialLandmarks:
        """Detect 68 facial landmarks within a face bounding box.

        Parameters
        ----------
        img_gray : ImageArray
            Grayscale input image.
        face_rect : FaceRect
            Face bounding box ``(x, y, w, h)`` from ``FaceDetector``.
        image_path : Path
            Path to the original image file (stored in result).

        Returns
        -------
        FacialLandmarks
            Dataclass with all landmark data.
        """
        x, y, w, h = face_rect
        dlib_rect = dlib.rectangle(
            left=int(x),
            top=int(y),
            right=int(x + w),
            bottom=int(y + h),
        )

        shape = self._predictor(img_gray, dlib_rect)
        landmarks_68 = [
            (shape.part(i).x, shape.part(i).y) for i in range(68)
        ]

        landmarks_by_region = self._group_by_region(landmarks_68)

        facial_landmarks = FacialLandmarks(
            image_path=image_path,
            face_rect=face_rect,
            landmarks_68=landmarks_68,
            landmarks_by_region=landmarks_by_region,
        )

        return facial_landmarks

    @staticmethod
    def check_pose(landmarks_68: LandmarksList) -> bool:
        """Check whether the face is in a frontal pose using symmetry heuristic.

        Uses the ratio of distances from the nose tip (landmark 30) to the
        outer corner of each eye (left eye: landmark 36, right eye: landmark 45).
        For a frontal face this ratio should be close to 1.0 (roughly 0.8-1.2).

        Parameters
        ----------
        landmarks_68 : LandmarksList
            Full list of 68 (x, y) landmark coordinates.

        Returns
        -------
        bool
            ``True`` if the face appears frontal, ``False`` if it appears
            non-frontal (head rotation >30°).
        """
        # Nose tip (landmark 30, 0-indexed)
        nose_tip = landmarks_68[30]
        # Left eye outer corner (landmark 36, 0-indexed)
        left_eye_outer = landmarks_68[36]
        # Right eye outer corner (landmark 45, 0-indexed)
        right_eye_outer = landmarks_68[45]

        dist_left = math.dist(nose_tip, left_eye_outer)
        dist_right = math.dist(nose_tip, right_eye_outer)

        if dist_right == 0:
            return True  # Degenerate case — assume frontal

        ratio = dist_left / dist_right

        # Frontal: ratio close to 1.0, allow 0.8-1.2 range
        # Non-frontal: ratio < 0.7 or > 1.3
        return 0.7 <= ratio <= 1.3

    @staticmethod
    def _group_by_region(landmarks_68: list[tuple[int, int]]) -> LandmarkRegions:
        """Group the 68 landmarks by facial region.

        Parameters
        ----------
        landmarks_68 : list of (int, int)
            Full list of 68 (x, y) landmark coordinates.

        Returns
        -------
        LandmarkRegions
            Dictionary mapping region names to lists of points.
        """
        return {
            name: [landmarks_68[i] for i in indices]
            for name, indices in REGION_INDICES.items()
        }
