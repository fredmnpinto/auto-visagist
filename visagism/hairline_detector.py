"""Estimate hairline position using Canny edge detection.

Applies Gaussian blur and Canny edge detection to a forehead context
region, then scans a narrow 1-pixel-wide center column for the first
strong edge from bottom to top.  Falls back to a geometric estimate
based on the superior-third = medium-third heuristic when no edge is
found.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

from visagism.constants import (
    HAIRLINE_CANNY_HIGH,
    HAIRLINE_CANNY_LOW,
    HAIRLINE_CLOSE_KSIZE,
    HAIRLINE_GAUSSIAN_KSIZE,
    HAIRLINE_ROI_UPWARD_EXPANSION,
)
from visagism.types import FacialLandmarks, ImageArray


class HairlineDetector:
    """Estimates the hairline y-coordinate using Canny edge detection.

    The detection works by:

    1. Computing the average eyebrow y-coordinate from brow landmarks.
    2. Defining a 1-pixel-wide search strip centred on the face midline,
       spanning from above the face top to just above the eyebrow line
       (excluding the bottom 25 % of the forehead).
    3. Extracting a wider forehead context (full face width) for Canny.
    4. Applying Gaussian blur and morphological closing to the context.
    5. Running Canny edge detection on the closed image.
    6. Sampling the centre column of the resulting edge map.
    7. Scanning bottom-to-top for the first edge pixel.
    8. Falling back to a geometric estimate if no edge is found.
    """

    def _compute_roi(
        self,
        img_gray: ImageArray,
        landmarks: FacialLandmarks,
    ) -> tuple[int, int, int, int, int, np.ndarray, np.ndarray]:
        """Compute forehead ROI and Canny context coordinates and extract regions.

        Parameters
        ----------
        img_gray : ImageArray
            Full grayscale input image.
        landmarks : FacialLandmarks
            Detected landmarks.

        Returns
        -------
        tuple
            ``(avg_eyebrow_y, x_start, x_end, y_start, y_end, roi, context)``
            where *roi* is the 1-pixel-wide centred strip and *context* is
            the wider forehead region used for Canny edge detection.
        """
        # 1. Compute average eyebrow y
        left_brow = landmarks.landmarks_by_region["left_eyebrow"]
        right_brow = landmarks.landmarks_by_region["right_eyebrow"]
        all_brow_pts = left_brow + right_brow  # 10 points total
        valid_brow_pts = [pt for pt in all_brow_pts if pt != (-1, -1)]

        face_rect = landmarks.face_rect
        fx, fy, fw, fh = face_rect

        if not valid_brow_pts:
            # Geometric fallback when all eyebrow points are missing
            avg_eyebrow_y = fy + int(fh * 0.20)
        else:
            avg_eyebrow_y = int(np.mean([pt[1] for pt in valid_brow_pts]))

        # 2. Define y-bounds for the search region
        upward_shift = int(fh * HAIRLINE_ROI_UPWARD_EXPANSION)
        y_start = fy - upward_shift
        forehead_height = avg_eyebrow_y - fy
        y_end = avg_eyebrow_y - int(forehead_height * 0.25)

        # 3. Define 1-pixel-wide ROI centred on face midline
        face_center_x = fx + fw // 2
        x_start = face_center_x
        x_end = face_center_x + 1

        # Clamp to image bounds
        img_h, img_w = img_gray.shape[:2]
        x_start = max(0, min(x_start, img_w - 1))
        x_end = max(0, min(x_end, img_w))
        y_start = max(0, min(y_start, img_h - 1))
        y_end = max(0, min(y_end, img_h))

        # 4. Extract 1-pixel ROI
        if y_start >= y_end or x_start >= x_end:
            roi = np.array([])
        else:
            roi = img_gray[y_start:y_end, x_start:x_end]

        # 5. Extract wider Canny context (full face width, same y-bounds)
        ctx_x_start = max(0, fx)
        ctx_x_end = min(fx + fw, img_w)
        ctx_y_start = y_start
        ctx_y_end = y_end

        if ctx_y_start >= ctx_y_end or ctx_x_start >= ctx_x_end:
            context = np.array([])
        else:
            context = img_gray[ctx_y_start:ctx_y_end, ctx_x_start:ctx_x_end]

        return (
            avg_eyebrow_y,
            x_start,
            x_end,
            y_start,
            y_end,
            roi,
            context,
        )

    def detect(
        self,
        img_gray: ImageArray,
        landmarks: FacialLandmarks,
        canny_low: int | None = None,
        canny_high: int | None = None,
        close_ksize: int | None = None,
        gaussian_ksize: int | None = None,
    ) -> dict:
        """Estimate hairline y-coordinate using Canny edge detection.

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
        canny_low : int or None, optional
            Lower threshold for Canny edge detection. If None, uses
            ``HAIRLINE_CANNY_LOW``.
        canny_high : int or None, optional
            Upper threshold for Canny edge detection. If None, uses
            ``HAIRLINE_CANNY_HIGH``.
        close_ksize : int or None, optional
            Kernel size for morphological closing. If None, uses
            ``HAIRLINE_CLOSE_KSIZE``.
        gaussian_ksize : int or None, optional
            Kernel size for Gaussian blur. If None, uses
            ``HAIRLINE_GAUSSIAN_KSIZE``.

        Returns
        -------
        dict
            Dictionary with 18 keys:
            - ``"hairline_y"``: Detected hairline y-coordinate (int)
            - ``"method"``: Detection method (str, ``"canny"`` or ``"fallback"``)
            - ``"roi_coords"``: Tuple (x_start, x_end, y_start, y_end)
            - ``"roi_raw"``: Raw 1-pixel forehead ROI (np.ndarray)
            - ``"canny_context_coords"``: Tuple (ctx_x_start, ctx_x_end,
              ctx_y_start, ctx_y_end)
            - ``"canny_context_raw"``: Raw Canny context (np.ndarray)
            - ``"closed_context"``: Image after Gaussian blur and morphological
              closing (np.ndarray).  Identical to ``"canny_context_raw"`` when
              closing is skipped.
            - ``"close_ksize"``: Morphological closing kernel size (int).  Zero
              means closing was skipped.
            - ``"canny_edge_map"``: Canny edge map (np.ndarray)
            - ``"center_column"``: Sampled centre column from edge map
              (np.ndarray)
            - ``"first_edge_idx"``: Index of first edge pixel in centre
              column when scanning bottom-to-top (int, -1 if none)
            - ``"edge_pixels_count"``: Total edge pixels in centre column
              (int)
            - ``"gaussian_ksize"``: Gaussian kernel size used (int)
            - ``"canny_low"``: Canny low threshold (int)
            - ``"canny_high"``: Canny high threshold (int)
            - ``"avg_eyebrow_y"``: Average eyebrow y-coordinate (int)
            - ``"face_rect"``: Face bounding box tuple
            - ``"searchable_rows"``: Number of rows scanned (int)
        """
        # Use provided parameters or fall back to constants
        canny_low_val = (
            canny_low if canny_low is not None else HAIRLINE_CANNY_LOW
        )
        canny_high_val = (
            canny_high if canny_high is not None else HAIRLINE_CANNY_HIGH
        )
        close_ksize_val = (
            close_ksize if close_ksize is not None else HAIRLINE_CLOSE_KSIZE
        )
        gaussian_ksize_val = (
            gaussian_ksize
            if gaussian_ksize is not None
            else HAIRLINE_GAUSSIAN_KSIZE
        )

        (
            avg_eyebrow_y,
            x_start,
            x_end,
            y_start,
            y_end,
            roi,
            context,
        ) = self._compute_roi(img_gray, landmarks)

        face_rect = landmarks.face_rect
        fx, fy, fw, fh = face_rect

        # Canny context coordinates
        ctx_x_start = max(0, fx)
        ctx_x_end = min(fx + fw, img_gray.shape[1])
        ctx_y_start = y_start
        ctx_y_end = y_end

        # Check if search region is valid
        searchable_rows = y_end - y_start if y_end > y_start else 0
        if (
            y_start >= y_end
            or searchable_rows < 2
            or x_start >= x_end
            or roi.size == 0
            or context.size == 0
        ):
            fallback_y = self._fallback(landmarks, avg_eyebrow_y)
            return {
                "hairline_y": fallback_y,
                "method": "fallback",
                "roi_coords": (x_start, x_end, y_start, y_end),
                "roi_raw": roi,
                "canny_context_coords": (
                    ctx_x_start,
                    ctx_x_end,
                    ctx_y_start,
                    ctx_y_end,
                ),
                "canny_context_raw": context,
                "closed_context": context,
                "close_ksize": 0,
                "canny_edge_map": np.array([]),
                "center_column": np.array([]),
                "first_edge_idx": -1,
                "edge_pixels_count": 0,
                "gaussian_ksize": gaussian_ksize_val,
                "canny_low": canny_low_val,
                "canny_high": canny_high_val,
                "avg_eyebrow_y": avg_eyebrow_y,
                "face_rect": face_rect,
                "searchable_rows": searchable_rows,
            }

        # Gaussian blur
        ksize = gaussian_ksize_val
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(context, (ksize, ksize), 0)

        # Morphological closing
        if close_ksize_val > 0:
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (close_ksize_val, close_ksize_val)
            )
            closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, close_kernel)
        else:
            closed = blurred

        # Canny edge detection
        edges = cv2.Canny(closed, canny_low_val, canny_high_val)

        # Sample centre column from edge map
        face_center_x = fx + fw // 2
        center_col_idx = face_center_x - ctx_x_start
        center_col_idx = max(0, min(center_col_idx, edges.shape[1] - 1))
        center_column = edges[:, center_col_idx]

        # Scan bottom-to-top for first edge pixel
        hairline_y = None
        method = "fallback"
        first_edge_idx = -1
        edge_pixels_count = int(np.count_nonzero(center_column == 255))

        for i in reversed(range(searchable_rows)):
            if center_column[i] == 255:
                hairline_y = y_start + i
                method = "canny"
                first_edge_idx = i
                break

        if hairline_y is None:
            warnings.warn(
                "Warning: No strong hairline edge detected in searchable region; "
                "using fallback estimate (superior third = medium third)."
            )
            hairline_y = self._fallback(landmarks, avg_eyebrow_y)

        return {
            "hairline_y": hairline_y,
            "method": method,
            "roi_coords": (x_start, x_end, y_start, y_end),
            "roi_raw": roi,
            "canny_context_coords": (
                ctx_x_start,
                ctx_x_end,
                ctx_y_start,
                ctx_y_end,
            ),
            "canny_context_raw": context,
            "closed_context": closed,
            "close_ksize": close_ksize_val,
            "canny_edge_map": edges,
            "center_column": center_column,
            "first_edge_idx": first_edge_idx,
            "edge_pixels_count": edge_pixels_count,
            "gaussian_ksize": ksize,
            "canny_low": canny_low_val,
            "canny_high": canny_high_val,
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
