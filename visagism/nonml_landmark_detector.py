"""Non-ML facial landmark detector using classical image processing.

Detects 14 anchor points (jaw extremes, eye corners, nose base/sides,
mouth corners, eyebrow peaks) using heuristic image-processing techniques.
The remaining 54 points in the 68-point dlib model are set to (-1, -1).

Detection methods adapted from classical computer vision literature
and OpenCV tutorials on facial feature detection.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from visagism.landmark_detector import LandmarkDetector
from visagism.types import (
    FaceRect,
    FacialLandmarks,
    ImageArray,
    LandmarksList,
)


class NonMLLandmarkDetector:
    """Detects 14 facial anchor points using classical image processing.

    Produces a 68-point landmark list compatible with the dlib 68-point
    model.  14 points are detected via heuristics; the remaining 54 are
    set to ``(-1, -1)``.

    Parameters
    ----------
    model_path : Path
        Path to the dlib model file (ignored, kept for interface
        compatibility).
    """

    def __init__(self, model_path: Path) -> None:
        """Initialise the non-ML landmark detector.

        Parameters
        ----------
        model_path : Path
            Path to the dlib shape predictor model file (ignored).
        """
        self._model_path = model_path

    def detect(
        self,
        img_gray: ImageArray,
        face_rect: FaceRect,
        image_path: Path,
    ) -> FacialLandmarks:
        """Detect 14 anchor landmarks within a face bounding box.

        Parameters
        ----------
        img_gray : ImageArray
            Grayscale input image.
        face_rect : FaceRect
            Face bounding box ``(x, y, w, h)``.
        image_path : Path
            Path to the original image file.

        Returns
        -------
        FacialLandmarks
            Dataclass with 68 landmarks (14 detected, 54 missing).
        """
        landmarks_68: LandmarksList = [(-1, -1)] * 68

        x, y, w, h = face_rect

        # Jawline extremes (indices 0, 8, 16)
        landmarks_68[0] = (x, y + int(h * 0.85))
        landmarks_68[8] = (x + w // 2, y + h)
        landmarks_68[16] = (x + w, y + int(h * 0.85))

        # Eyes (indices 36, 39, 42, 45)
        eye_points = self._detect_eyes(img_gray, face_rect)
        landmarks_68[36] = eye_points[0]
        landmarks_68[39] = eye_points[1]
        landmarks_68[42] = eye_points[2]
        landmarks_68[45] = eye_points[3]

        # Nose (indices 31, 33, 35)
        nose_points = self._detect_nose(img_gray, face_rect)
        landmarks_68[31] = nose_points[0]
        landmarks_68[33] = nose_points[1]
        landmarks_68[35] = nose_points[2]

        # Mouth (indices 48, 54)
        mouth_points = self._detect_mouth(img_gray, face_rect)
        landmarks_68[48] = mouth_points[0]
        landmarks_68[54] = mouth_points[1]

        # Eyebrows (indices 19, 24)
        brow_points = self._detect_eyebrows(img_gray, face_rect)
        landmarks_68[19] = brow_points[0]
        landmarks_68[24] = brow_points[1]

        landmarks_by_region = LandmarkDetector._group_by_region(landmarks_68)

        return FacialLandmarks(
            image_path=image_path,
            face_rect=face_rect,
            landmarks_68=landmarks_68,
            landmarks_by_region=landmarks_by_region,
        )

    def _detect_eyes(
        self,
        img_gray: ImageArray,
        face_rect: FaceRect,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Detect eye corners using intensity blob detection.

        Parameters
        ----------
        img_gray : ImageArray
            Full grayscale input image.
        face_rect : FaceRect
            Face bounding box ``(x, y, w, h)``.

        Returns
        -------
        tuple of 4 (int, int)
            Left-eye left corner, left-eye right corner,
            right-eye left corner, right-eye right corner.
        """
        x, y, w, h = face_rect

        # Upper-face ROI: top 45 % of face
        y_end = y + int(h * 0.45)
        if y_end <= y or y >= img_gray.shape[0]:
            return self._eye_fallback(x, y, w, h)

        roi = img_gray[y:y_end, x:x + w]
        if roi.size == 0:
            return self._eye_fallback(x, y, w, h)

        # Threshold for dark regions (eyes are darker than skin)
        # Use Otsu when there is enough contrast; otherwise fixed.
        if roi.max() - roi.min() > 20:
            _, thresh = cv2.threshold(
                roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            thresh = cv2.inRange(roi, 0, 80)

        # Split into left/right halves
        mid_x = w // 2
        left_thresh = thresh[:, :mid_x]
        right_thresh = thresh[:, mid_x:]

        left_eye = self._find_eye_blob(left_thresh, x, y)
        right_eye = self._find_eye_blob(right_thresh, x + mid_x, y)

        return left_eye[0], left_eye[1], right_eye[0], right_eye[1]

    def _eye_fallback(
        self, x: int, y: int, w: int, h: int
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Return geometric eye-corner estimates.

        Parameters
        ----------
        x, y, w, h : int
            Face bounding box.

        Returns
        -------
        tuple of 4 (int, int)
            Estimated eye corners.
        """
        eye_y = y + int(h * 0.25)
        left_l = x + int(w * 0.20)
        left_r = x + int(w * 0.35)
        right_l = x + int(w * 0.65)
        right_r = x + int(w * 0.80)
        return (left_l, eye_y), (left_r, eye_y), (right_l, eye_y), (right_r, eye_y)

    def _find_eye_blob(
        self,
        thresh: npt.NDArray[np.uint8],
        offset_x: int,
        offset_y: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Find the largest dark blob and return its left/right edges.

        Parameters
        ----------
        thresh : np.ndarray
            Binary thresholded ROI.
        offset_x : int
            X offset of the ROI in the full image.
        offset_y : int
            Y offset of the ROI in the full image.

        Returns
        -------
        tuple of 2 (int, int)
            Left and right eye corners.
        """
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = max(thresh.shape[0] * thresh.shape[1] * 0.001, 10.0)
        best_contour = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area <= best_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter ** 2)
                if circularity > 0.2:  # Allow slightly elongated shapes
                    best_area = area
                    best_contour = cnt

        if best_contour is not None:
            bx, by, bw, bh = cv2.boundingRect(best_contour)
            cy = offset_y + by + bh // 2
            return (offset_x + bx, cy), (offset_x + bx + bw, cy)

        # Fallback: geometric estimate centred in the ROI
        cx = offset_x + thresh.shape[1] // 2
        cy = offset_y + thresh.shape[0] // 2
        w_est = max(thresh.shape[1] // 4, 5)
        return (cx - w_est // 2, cy), (cx + w_est // 2, cy)

    def _detect_nose(
        self,
        img_gray: ImageArray,
        face_rect: FaceRect,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Detect nose base and sides using brightness profile.

        Parameters
        ----------
        img_gray : ImageArray
            Full grayscale input image.
        face_rect : FaceRect
            Face bounding box ``(x, y, w, h)``.

        Returns
        -------
        tuple of 3 (int, int)
            Left nostril, nose tip/base, right nostril.
        """
        x, y, w, h = face_rect

        # Nose ROI: middle 30 % of face height, centred horizontally
        y_start = y + int(h * 0.35)
        y_end = y + int(h * 0.75)
        cx = x + w // 2

        roi = img_gray[y_start:y_end, x:x + w]
        if roi.size == 0:
            nose_y = y + int(h * 0.60)
            return (
                (cx - int(w * 0.08), nose_y),
                (cx, nose_y),
                (cx + int(w * 0.08), nose_y),
            )

        # Vertical brightness profile on centreline
        col_idx = cx - x
        col_idx = max(0, min(col_idx, roi.shape[1] - 1))
        profile = roi[:, col_idx].astype(np.float32)

        # 33 = lowest bright point in lower half (nose base)
        lower_half_start = len(profile) // 2
        lower_profile = profile[lower_half_start:]
        if len(lower_profile) > 0:
            local_max_idx = int(np.argmax(lower_profile))
            nose_base_y = y_start + lower_half_start + local_max_idx
        else:
            nose_base_y = y + int(h * 0.60)

        # At nose base level, find dark valleys left and right (nostrils)
        row_idx = nose_base_y - y_start
        row_idx = max(0, min(row_idx, roi.shape[0] - 1))
        row = roi[row_idx, :].astype(np.float32)

        # Left valley: minimum brightness left of centre
        left_half = row[:col_idx]
        if len(left_half) > 0:
            left_valley = int(np.argmin(left_half))
            nose_left_x = x + left_valley
        else:
            nose_left_x = cx - int(w * 0.08)

        # Right valley: minimum brightness right of centre
        right_half = row[col_idx + 1:]
        if len(right_half) > 0:
            right_valley = int(np.argmin(right_half))
            nose_right_x = x + col_idx + 1 + right_valley
        else:
            nose_right_x = cx + int(w * 0.08)

        return (
            (nose_left_x, nose_base_y),
            (cx, nose_base_y),
            (nose_right_x, nose_base_y),
        )

    def _detect_mouth(
        self,
        img_gray: ImageArray,
        face_rect: FaceRect,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Detect mouth corners using horizontal gradient.

        Parameters
        ----------
        img_gray : ImageArray
            Full grayscale input image.
        face_rect : FaceRect
            Face bounding box ``(x, y, w, h)``.

        Returns
        -------
        tuple of 2 (int, int)
            Left and right mouth corners.
        """
        x, y, w, h = face_rect

        # Lower-third ROI
        y_start = y + int(h * 0.60)
        roi = img_gray[y_start:y + h, x:x + w]

        if roi.size == 0:
            mouth_y = y + int(h * 0.80)
            return (x + int(w * 0.25), mouth_y), (x + int(w * 0.75), mouth_y)

        # Horizontal Sobel gradient
        sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)

        # Sum gradients vertically to get a 1-D profile
        gradient_profile = np.sum(np.abs(sobel_x), axis=0)

        # Smooth the profile
        if len(gradient_profile) > 5:
            kernel = np.ones(5) / 5
            gradient_profile = np.convolve(
                gradient_profile, kernel, mode="same"
            )

        mid = len(gradient_profile) // 2
        left_profile = gradient_profile[:mid]
        right_profile = gradient_profile[mid:]

        if len(left_profile) > 0:
            left_idx = int(np.argmax(left_profile))
        else:
            left_idx = w // 4

        if len(right_profile) > 0:
            right_idx = mid + int(np.argmax(right_profile))
        else:
            right_idx = 3 * w // 4

        mouth_y = y_start + roi.shape[0] // 2

        return (x + left_idx, mouth_y), (x + right_idx, mouth_y)

    def _detect_eyebrows(
        self,
        img_gray: ImageArray,
        face_rect: FaceRect,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Detect eyebrow peaks using horizontal edge detection.

        Parameters
        ----------
        img_gray : ImageArray
            Full grayscale input image.
        face_rect : FaceRect
            Face bounding box ``(x, y, w, h)``.

        Returns
        -------
        tuple of 2 (int, int)
            Left and right eyebrow peaks.
        """
        x, y, w, h = face_rect

        # Eyebrow ROIs: roughly top 15-40 % of face
        y_start = y + int(h * 0.10)
        y_end = y + int(h * 0.40)
        roi = img_gray[y_start:y_end, x:x + w]

        if roi.size == 0:
            brow_y = y + int(h * 0.20)
            return (x + int(w * 0.30), brow_y), (x + int(w * 0.70), brow_y)

        # Horizontal edges (Sobel Y)
        sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

        # Split into left/right halves
        mid_x = w // 2
        left_sobel = sobel_y[:, :mid_x]
        right_sobel = sobel_y[:, mid_x:]

        left_peak = self._find_brow_peak(left_sobel, x, y_start)
        right_peak = self._find_brow_peak(right_sobel, x + mid_x, y_start)

        return left_peak, right_peak

    def _find_brow_peak(
        self,
        sobel_y: npt.NDArray[np.float64],
        offset_x: int,
        offset_y: int,
    ) -> tuple[int, int]:
        """Find the highest strong horizontal edge in the ROI.

        Parameters
        ----------
        sobel_y : np.ndarray
            Vertical Sobel response.
        offset_x : int
            X offset of the ROI in the full image.
        offset_y : int
            Y offset of the ROI in the full image.

        Returns
        -------
        tuple[int, int]
            Peak eyebrow point.
        """
        abs_sobel = np.abs(sobel_y)

        max_val = float(np.max(abs_sobel)) if abs_sobel.size > 0 else 0.0
        threshold = max_val * 0.3 if max_val > 0 else 1.0

        strong_y, strong_x = np.where(abs_sobel > threshold)

        if len(strong_y) > 0:
            # Highest edge point = minimum y
            min_y_idx = int(np.argmin(strong_y))
            peak_y = offset_y + strong_y[min_y_idx]
            peak_x = offset_x + strong_x[min_y_idx]
            return (peak_x, peak_y)

        # Fallback: geometric estimate
        cx = offset_x + sobel_y.shape[1] // 2
        cy = offset_y + sobel_y.shape[0] // 4
        return (cx, cy)
