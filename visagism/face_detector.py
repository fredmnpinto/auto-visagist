"""Face detection using OpenCV Haar cascade classifier."""

from __future__ import annotations

import os

import cv2

from visagism.constants import (
    BBOX_EXPANSION_FACTOR,
    HAAR_MIN_FACE_SIZE,
    HAAR_MIN_NEIGHBORS,
    HAAR_SCALE_FACTOR,
    SMALL_FACE_AREA_THRESHOLD,
)
from visagism.errors import NoFaceDetectedError
from visagism.types import FaceRect, ImageArray


def _locate_haar_cascade() -> str:
    """Locate the Haar cascade XML file on disk.

    Searches in the following locations:
    1. ``cv2.data.haarcascades`` (OpenCV >= 4.5.1 with ``opencv-python``)
    2. Relative to ``cv2.__file__`` in the nix store structure
    3. Common system-wide installation paths

    Returns
    -------
    str
        Absolute path to ``haarcascade_frontalface_default.xml``.

    Raises
    ------
    FileNotFoundError
        If the cascade file cannot be found.
    """
    # Strategy 1: cv2.data.haarcascades (opencv-python packages)
    try:
        candidate = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(candidate):
            return candidate
    except AttributeError:
        pass

    cv2_dir = os.path.dirname(cv2.__file__)

    # Strategy 2: nix store relative path
    # cv2.__file__ is at:
    #   /nix/store/<hash>-opencv-<ver>/lib/python3.12/site-packages/cv2/__init__.py
    # cascade is at:
    #   /nix/store/<hash>-opencv-<ver>/share/opencv4/haarcascades/haarcascade_frontalface_default.xml
    parts = cv2_dir.split(os.sep)
    for i, p in enumerate(parts):
        if p == "store":
            store_root = os.sep.join(parts[: i + 2])
            candidate = os.path.join(
                store_root,
                "share",
                "opencv4",
                "haarcascades",
                "haarcascade_frontalface_default.xml",
            )
            if os.path.exists(candidate):
                return candidate
            break

    # Strategy 3: walk up from site-packages looking for share/opencv4
    site_packages = os.path.dirname(cv2_dir)
    search_root = site_packages
    for _ in range(4):
        for subdir in [
            "share/opencv4/haarcascades",
            "etc/opencv4/haarcascades",
            "etc/haarcascades",
            "data/haarcascades",
        ]:
            candidate = os.path.join(
                search_root, subdir, "haarcascade_frontalface_default.xml"
            )
            if os.path.exists(candidate):
                return candidate
        search_root = os.path.dirname(search_root)

    raise FileNotFoundError(
        "Could not locate haarcascade_frontalface_default.xml. "
        "Ensure OpenCV is properly installed."
    )


class FaceDetector:
    """Detects human faces in images using OpenCV Haar cascade.

    Uses the built-in ``haarcascade_frontalface_default.xml`` cascade.
    If multiple faces are detected, the largest one (by area) is selected.
    The bounding box is expanded by 20% to provide context.
    """

    def __init__(self) -> None:
        """Initialise the face detector with a Haar cascade classifier."""
        cascade_path = _locate_haar_cascade()
        self._cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, img_gray: ImageArray, img_shape: tuple[int, int]) -> FaceRect:
        """Detect a face in a grayscale image.

        Parameters
        ----------
        img_gray : ImageArray
            Grayscale input image.
        img_shape : tuple of int
            Image dimensions ``(height, width)``.

        Returns
        -------
        FaceRect
            Bounding box ``(x, y, width, height)`` of the detected face,
            expanded by 20%.

        Raises
        ------
        NoFaceDetectedError
            If no face is found in the image.
        """
        faces = self._cascade.detectMultiScale(
            img_gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_FACE_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # detectMultiScale returns ndarray of shape (N, 4) or empty tuple
        if len(faces) == 0:
            raise NoFaceDetectedError(
                "No face detected. Please provide a clear frontal photo."
            )

        # Select largest face by area
        if len(faces) > 1:
            areas = faces[:, 2] * faces[:, 3]
            best = faces[areas.argmax()]
            print("Multiple faces detected. Analyzing the largest face.")
        else:
            best = faces[0]

        x, y, w, h = int(best[0]), int(best[1]), int(best[2]), int(best[3])

        # Check if face is small relative to image and warn
        img_area = img_shape[0] * img_shape[1]
        if (w * h) < (img_area * SMALL_FACE_AREA_THRESHOLD):
            print(
                "Warning: Detected face is small; "
                "consider using a closer photo."
            )

        # Expand bounding box by 20%
        dx = int(w * BBOX_EXPANSION_FACTOR / 2)
        dy = int(h * BBOX_EXPANSION_FACTOR / 2)
        img_h, img_w = img_shape

        x_new = max(0, x - dx)
        y_new = max(0, y - dy)
        w_new = min(img_w - x_new, w + 2 * dx)
        h_new = min(img_h - y_new, h + 2 * dy)

        return (x_new, y_new, w_new, h_new)
