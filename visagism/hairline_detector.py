"""Estimate hairline position using vertical intensity gradient analysis.

Uses CLAHE contrast enhancement and vertical intensity gradient scanning
above the eyebrows to locate the hairline.  The algorithm looks for the
steepest intensity change (bright skin → dark hair) in the forehead ROI.
Falls back to a geometric estimate based on the
superior-third = medium-third heuristic.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

from visagism.constants import (
    HAIRLINE_CLAHE_CLIP,
    HAIRLINE_CLAHE_GRID,
    HAIRLINE_MIN_GRADIENT_RATIO,
)
from visagism.types import FacialLandmarks, ImageArray


class HairlineDetector:
    """Estimates the hairline y-coordinate using vertical intensity gradient.

    The detection works by:

    1. Computing the average eyebrow y-coordinate from brow landmarks.
    2. Defining a search region between the top of the face bounding box
       and the eyebrow line.
    3. Applying CLAHE contrast enhancement on the forehead ROI.
    4. Computing the average intensity per row and its vertical gradient.
    5. Finding the row with the steepest intensity change.
    6. Validating that the gradient is significantly above the median.
    7. Falling back to a geometric estimate if no strong gradient is found.
    """

    def _compute_roi(
        self,
        img_gray: ImageArray,
        landmarks: FacialLandmarks,
    ) -> tuple[int, int, int, int, int, np.ndarray]:
        """Compute forehead ROI coordinates and extract the ROI.

        Parameters
        ----------
        img_gray : ImageArray
            Full grayscale input image.
        landmarks : FacialLandmarks
            Detected landmarks.

        Returns
        -------
        tuple
            ``(avg_eyebrow_y, x_start, x_end, y_start, y_end, roi)`` where
            *roi* is the extracted grayscale forehead region.
        """
        # 1. Compute average eyebrow y
        left_brow = landmarks.landmarks_by_region["left_eyebrow"]
        right_brow = landmarks.landmarks_by_region["right_eyebrow"]
        all_brow_pts = left_brow + right_brow  # 10 points total
        avg_eyebrow_y = int(np.mean([pt[1] for pt in all_brow_pts]))

        face_rect = landmarks.face_rect
        fx, fy, fw, fh = face_rect

        # 2. Define search region (forehead area within face bounding box)
        x_start = fx
        x_end = fx + fw
        y_start = fy
        y_end = avg_eyebrow_y

        # Clamp to image bounds
        img_h, img_w = img_gray.shape[:2]
        x_start = max(0, min(x_start, img_w - 1))
        x_end = max(0, min(x_end, img_w))
        y_start = max(0, min(y_start, img_h - 1))
        y_end = max(0, min(y_end, img_h))

        # 3. Extract forehead ROI
        if y_start >= y_end or x_start >= x_end:
            roi = np.array([])
        else:
            roi = img_gray[y_start:y_end, x_start:x_end]

        return avg_eyebrow_y, x_start, x_end, y_start, y_end, roi

    def detect(
        self,
        img_gray: ImageArray,
        landmarks: FacialLandmarks,
    ) -> dict:
        """Estimate hairline y-coordinate using vertical intensity gradient.

        Runs the full detection pipeline and returns all intermediate data
        alongside the final result so that the algorithm is the single
        source of truth for both production and debug use.

        Parameters
        ----------
        img_gray : ImageArray
            Full grayscale input image.
        landmarks : FacialLandmarks
            Detected landmarks with ``face_rect``, ``landmarks_by_region``,
            and optionally ``hairline_y``.

        Returns
        -------
        dict
            Dictionary with 16 keys:
            - ``"roi_raw"``: Raw forehead ROI (np.ndarray)
            - ``"roi_enhanced"``: CLAHE-enhanced ROI (np.ndarray)
            - ``"row_intensities"``: Average intensity per row (np.ndarray)
            - ``"gradient"``: Signed gradient per row (np.ndarray)
            - ``"abs_gradient"``: Absolute gradient per row (np.ndarray)
            - ``"max_gradient_idx"``: Index of max gradient (int)
            - ``"max_gradient_value"``: Value of max gradient (float)
            - ``"max_gradient_value_full"``: Max over full gradient array (float)
            - ``"median_gradient"``: Median absolute gradient (float)
            - ``"gradient_ratio"``: Max / median ratio (float)
            - ``"hairline_y"``: Detected hairline y-coordinate (int)
            - ``"method"``: Detection method (str, ``"edge"`` or ``"fallback"``)
            - ``"roi_coords"``: Tuple (x_start, x_end, y_start, y_end)
            - ``"avg_eyebrow_y"``: Average eyebrow y-coordinate (int)
            - ``"face_rect"``: Face bounding box tuple
            - ``"searchable_rows"``: Number of rows in the searchable region (int)
        """
        avg_eyebrow_y, x_start, x_end, y_start, y_end, roi = self._compute_roi(
            img_gray, landmarks
        )

        face_rect = landmarks.face_rect

        # Check if search region is valid
        if (y_start >= y_end or (y_end - y_start) < 2
                or x_start >= x_end or roi.size == 0):
            fallback_y = self._fallback(landmarks, avg_eyebrow_y)
            return {
                "roi_raw": roi,
                "roi_enhanced": np.array([]),
                "row_intensities": np.array([]),
                "gradient": np.array([]),
                "abs_gradient": np.array([]),
                "max_gradient_idx": -1,
                "max_gradient_value": 0.0,
                "max_gradient_value_full": 0.0,
                "median_gradient": 0.0,
                "gradient_ratio": 0.0,
                "hairline_y": fallback_y,
                "method": "fallback",
                "roi_coords": (x_start, x_end, y_start, y_end),
                "avg_eyebrow_y": avg_eyebrow_y,
                "face_rect": face_rect,
                "searchable_rows": 0,
            }

        # CLAHE enhancement
        clahe = cv2.createCLAHE(
            clipLimit=HAIRLINE_CLAHE_CLIP,
            tileGridSize=HAIRLINE_CLAHE_GRID,
        )
        roi_enhanced = clahe.apply(roi)

        # Compute average intensity per row
        row_intensities = np.mean(roi_enhanced, axis=1)

        # Exclude bottom 25% of ROI to avoid eyebrows
        roi_h = roi_enhanced.shape[0]
        exclude_bottom = int(roi_h * 0.25)
        searchable_rows = roi_h - exclude_bottom

        if searchable_rows < 2:
            fallback_y = self._fallback(landmarks, avg_eyebrow_y)
            return {
                "roi_raw": roi,
                "roi_enhanced": roi_enhanced,
                "row_intensities": row_intensities,
                "gradient": np.array([]),
                "abs_gradient": np.array([]),
                "max_gradient_idx": -1,
                "max_gradient_value": 0.0,
                "max_gradient_value_full": 0.0,
                "median_gradient": 0.0,
                "gradient_ratio": 0.0,
                "hairline_y": fallback_y,
                "method": "fallback",
                "roi_coords": (x_start, x_end, y_start, y_end),
                "avg_eyebrow_y": avg_eyebrow_y,
                "face_rect": face_rect,
                "searchable_rows": searchable_rows,
            }

        # Compute signed vertical gradient
        gradient = np.gradient(row_intensities)

        # Only search top 75% of ROI
        search_gradient = gradient[:searchable_rows]
        search_intensities = row_intensities[:searchable_rows]

        # Find median of absolute gradient for threshold
        abs_gradient = np.abs(gradient)
        search_abs_gradient = np.abs(search_gradient)
        median_gradient = float(np.median(abs_gradient))
        if median_gradient == 0:
            median_gradient = 1e-6

        threshold = median_gradient * HAIRLINE_MIN_GRADIENT_RATIO

        # Search from top down for first strong positive gradient
        mean_intensity = float(np.mean(row_intensities))
        hairline_y = None
        method = "fallback"
        for i in range(len(search_gradient)):
            if (search_gradient[i] > threshold
                    and search_intensities[i] < mean_intensity):
                hairline_y = y_start + i
                method = "edge"
                break

        if hairline_y is None:
            warnings.warn(
                "Warning: No strong hairline edge detected in searchable region; "
                "using fallback estimate (superior third = medium third)."
            )
            hairline_y = self._fallback(landmarks, avg_eyebrow_y)

        max_gradient_idx = int(np.argmax(search_abs_gradient))
        max_gradient_value = float(search_abs_gradient[max_gradient_idx])
        max_gradient_value_full = float(np.max(abs_gradient))
        gradient_ratio = max_gradient_value / median_gradient

        return {
            "roi_raw": roi,
            "roi_enhanced": roi_enhanced,
            "row_intensities": row_intensities,
            "gradient": gradient,
            "abs_gradient": abs_gradient,
            "max_gradient_idx": max_gradient_idx,
            "max_gradient_value": max_gradient_value,
            "max_gradient_value_full": max_gradient_value_full,
            "median_gradient": median_gradient,
            "gradient_ratio": gradient_ratio,
            "hairline_y": hairline_y,
            "method": method,
            "roi_coords": (x_start, x_end, y_start, y_end),
            "avg_eyebrow_y": avg_eyebrow_y,
            "face_rect": face_rect,
            "searchable_rows": searchable_rows,
        }

    def _fallback(
        self,
        landmarks: FacialLandmarks,
        avg_eyebrow_y: int,
    ) -> int:
        """Geometric fallback: superior third = medium third.

        Uses the heuristic that the superior third (hairline to brows)
        equals the medium third (brows to nose base).

        Parameters
        ----------
        landmarks : FacialLandmarks
            Detected landmark data (used for face_rect and landmarks_68).
        avg_eyebrow_y : int
            Average y-coordinate of the eyebrows.

        Returns
        -------
        int
            Estimated hairline y-coordinate, clamped to the face box top.
        """
        nose_base_y = landmarks.landmarks_68[33][1]

        medium_third_height = nose_base_y - avg_eyebrow_y
        hairline_y = avg_eyebrow_y - medium_third_height

        # Clamp to face box top
        hairline_y = max(hairline_y, landmarks.face_rect[1])
        return hairline_y
