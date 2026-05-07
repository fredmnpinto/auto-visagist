"""Load and validate input images for facial analysis.

Handles format validation, resolution checks, alpha channel conversion,
and grayscale conversion.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from visagism.constants import MIN_RESOLUTION, SUPPORTED_FORMATS
from visagism.errors import CorruptedImageError, ImageError, UnsupportedFormatError
from visagism.types import ImageArray


class ImageLoader:
    """Loads and validates input images.

    Provides a single static method ``load()`` that validates the file
    path, format, readability, and resolution, then returns the image
    as both BGR and grayscale arrays.
    """

    @staticmethod
    def load(path: Path) -> tuple[ImageArray, ImageArray]:
        """Load and validate an image from the given file path.

        Performs the following validations:

        * File exists (raises ``ImageError``)
        * Extension is in ``SUPPORTED_FORMATS`` (raises ``UnsupportedFormatError``)
        * Image can be decoded (raises ``CorruptedImageError``)
        * Resolution below minimum recommendation prints a warning

        If the image has an alpha channel (BGRA), it is converted to BGR.

        Parameters
        ----------
        path : Path
            Path to the input image file.

        Returns
        -------
        tuple of ImageArray
            A pair ``(img_bgr, img_gray)`` where ``img_bgr`` is the BGR image
            and ``img_gray`` is the grayscale version.

        Raises
        ------
        ImageError
            If the file does not exist.
        UnsupportedFormatError
            If the file extension is not a supported format.
        CorruptedImageError
            If the image file cannot be read or is corrupted.
        """
        # Check file exists
        if not path.exists():
            raise ImageError(f"File not found: {path}")

        # Check format
        ext = path.suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported format: {ext}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        # Read image - cv2.imread returns None on failure silently
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            # Try reading with unchanged flag to handle alpha channels
            img_any = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img_any is None:
                raise CorruptedImageError(
                    f"Cannot read image file: {path}. File may be corrupted"
                )
            # Convert BGRA to BGR if alpha channel present
            if img_any.shape[2] == 4:
                img_bgr = cv2.cvtColor(img_any, cv2.COLOR_BGRA2BGR)
            else:
                img_bgr = img_any

        # Check minimum resolution
        height, width = img_bgr.shape[:2]
        min_w, min_h = MIN_RESOLUTION
        if width < min_w or height < min_h:
            print(
                f"Warning: Image resolution low ({width}x{height}). "
                f"Recommended minimum: {min_w}x{min_h} pixels."
            )

        # Convert to grayscale
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        return img_bgr, img_gray
