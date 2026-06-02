"""Non-ML facial landmark detector using classical image processing.

Detects 14 anchor points (jaw extremes, eye corners, nose base/sides,
mouth corners, eyebrow peaks) using heuristic image-processing techniques.
The remaining 54 points in the 68-point dlib model are set to (-1, -1).

Detection methods adapted from classical computer vision literature
and OpenCV tutorials on facial feature detection.
"""

from __future__ import annotations

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
        eye_points, _ = self._detect_eyes(img_gray, face_rect)
        landmarks_68[36] = eye_points[0]
        landmarks_68[39] = eye_points[1]
        landmarks_68[42] = eye_points[2]
        landmarks_68[45] = eye_points[3]

        # Nose (indices 31, 33, 35)
        nose_points, _ = self._detect_nose(img_gray, face_rect)
        landmarks_68[31] = nose_points[0]
        landmarks_68[33] = nose_points[1]
        landmarks_68[35] = nose_points[2]

        # Mouth (indices 48, 54)
        mouth_points, _ = self._detect_mouth(img_gray, face_rect)
        landmarks_68[48] = mouth_points[0]
        landmarks_68[54] = mouth_points[1]

        # Eyebrows (indices 19, 24)
        brow_points, _ = self._detect_eyebrows(img_gray, face_rect)
        landmarks_68[19] = brow_points[0]
        landmarks_68[24] = brow_points[1]

        landmarks_by_region = LandmarkDetector._group_by_region(landmarks_68)

        return FacialLandmarks(
            image_path=image_path,
            face_rect=face_rect,
            landmarks_68=landmarks_68,
            landmarks_by_region=landmarks_by_region,
        )

    def detect_with_steps(
        self,
        img_gray: ImageArray,
        face_rect: FaceRect,
        image_path: Path,
    ) -> tuple[FacialLandmarks, dict]:
        """Detect 14 anchor landmarks and return intermediate step data.

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
        tuple[FacialLandmarks, dict]
            Detected landmarks and a dictionary with diagnostic data
            for each step.
        """
        landmarks_68: LandmarksList = [(-1, -1)] * 68
        x, y, w, h = face_rect
        steps: dict = {"face_rect": face_rect}

        # Jawline extremes (indices 0, 8, 16)
        landmarks_68[0] = (x, y + int(h * 0.85))
        landmarks_68[8] = (x + w // 2, y + h)
        landmarks_68[16] = (x + w, y + int(h * 0.85))

        # Eyes (indices 36, 39, 42, 45)
        eye_points, eye_steps = self._detect_eyes(img_gray, face_rect)
        landmarks_68[36] = eye_points[0]
        landmarks_68[39] = eye_points[1]
        landmarks_68[42] = eye_points[2]
        landmarks_68[45] = eye_points[3]
        steps["eye"] = eye_steps

        # Nose (indices 31, 33, 35)
        nose_points, nose_steps = self._detect_nose(img_gray, face_rect)
        landmarks_68[31] = nose_points[0]
        landmarks_68[33] = nose_points[1]
        landmarks_68[35] = nose_points[2]
        steps["nose"] = nose_steps

        # Mouth (indices 48, 54)
        mouth_points, mouth_steps = self._detect_mouth(img_gray, face_rect)
        landmarks_68[48] = mouth_points[0]
        landmarks_68[54] = mouth_points[1]
        steps["mouth"] = mouth_steps

        # Eyebrows (indices 19, 24)
        brow_points, brow_steps = self._detect_eyebrows(img_gray, face_rect)
        landmarks_68[19] = brow_points[0]
        landmarks_68[24] = brow_points[1]
        steps["eyebrow"] = brow_steps

        landmarks_by_region = LandmarkDetector._group_by_region(landmarks_68)

        landmarks = FacialLandmarks(
            image_path=image_path,
            face_rect=face_rect,
            landmarks_68=landmarks_68,
            landmarks_by_region=landmarks_by_region,
        )

        steps = {
            "face_rect": face_rect,
            "eye": eye_steps,
            "eye_left_roi": eye_steps.get("left_roi"),
            "eye_right_roi": eye_steps.get("right_roi"),
            "eye_left_pupil": eye_steps.get("left_pupil"),
            "eye_right_pupil": eye_steps.get("right_pupil"),
            "eye_left_corners": eye_steps.get("left_corners"),
            "eye_right_corners": eye_steps.get("right_corners"),
            "nose": nose_steps,
            "nose_roi": nose_steps.get("roi"),
            "nose_profile": nose_steps.get("profile"),
            "mouth": mouth_steps,
            "mouth_roi": mouth_steps.get("roi"),
            "mouth_gradient": mouth_steps.get("gradient_profile"),
            "eyebrow": brow_steps,
            "eyebrow_roi": brow_steps.get("roi"),
            "eyebrow_edges": brow_steps.get("sobel_y"),
        }

        return landmarks, steps

    def _detect_eyes(self, img_gray, face_rect):
        x, y, w, h = face_rect

        # Tight eye ROIs using face proportions
        left_roi_x = x + int(w * 0.15)
        left_roi_y = y + int(h * 0.35)  # 35% down from top
        left_roi_w = int(w * 0.30)
        left_roi_h = int(h * 0.20)      # Keep 20% height (now ends at 45%)

        right_roi_x = x + int(w * 0.55)
        right_roi_y = y + int(h * 0.35)  # 35% down from top
        right_roi_w = int(w * 0.30)
        right_roi_h = int(h * 0.20)      # Keep 20% height (now ends at 45%)

        # Extract ROIs
        left_roi = img_gray[
            left_roi_y:left_roi_y + left_roi_h,
            left_roi_x:left_roi_x + left_roi_w,
        ]
        right_roi = img_gray[
            right_roi_y:right_roi_y + right_roi_h,
            right_roi_x:right_roi_x + right_roi_w,
        ]

        # Find pupils (darkest points)
        left_pupil = self._find_pupil(left_roi, left_roi_x, left_roi_y)
        right_pupil = self._find_pupil(right_roi, right_roi_x, right_roi_y)

        # Find corners by scanning from pupil
        left_corners = self._find_eye_corners(
            left_roi, left_pupil, left_roi_x, left_roi_y
        )
        right_corners = self._find_eye_corners(
            right_roi, right_pupil, right_roi_x, right_roi_y
        )

        step_data = {
            "left_roi": left_roi,
            "right_roi": right_roi,
            "left_pupil": left_pupil,
            "right_pupil": right_pupil,
            "left_corners": left_corners,
            "right_corners": right_corners,
        }

        return (
            left_corners[0], left_corners[1],
            right_corners[0], right_corners[1],
        ), step_data

    def _find_pupil(self, roi, offset_x, offset_y):
        if roi.size == 0:
            return (offset_x + roi.shape[1] // 2, offset_y + roi.shape[0] // 2)

        min_val = np.min(roi)
        threshold = min_val + (np.max(roi) - min_val) * 0.1
        dark_mask = roi <= threshold
        dark_y, dark_x = np.where(dark_mask)

        if len(dark_y) > 0:
            pupil_x = int(np.mean(dark_x)) + offset_x
            pupil_y = int(np.mean(dark_y)) + offset_y
            return (pupil_x, pupil_y)

        return (offset_x + roi.shape[1] // 2, offset_y + roi.shape[0] // 2)

    def _find_eye_corners(self, roi, pupil, offset_x, offset_y):
        if roi.size == 0:
            return (pupil[0], pupil[1]), (pupil[0], pupil[1])

        px = pupil[0] - offset_x
        py = pupil[1] - offset_y

        # Clamp to ROI bounds
        px = max(0, min(px, roi.shape[1] - 1))
        py = max(0, min(py, roi.shape[0] - 1))

        # Get horizontal row through pupil
        row = roi[py, :].astype(np.float32)

        # Smooth
        if len(row) > 5:
            row = cv2.GaussianBlur(row.reshape(-1, 1), (5, 1), 0).flatten()

        # Find left corner: scan left, find brightness jump
        left_x = px
        for i in range(px, 0, -1):
            if row[i] > row[i-1] + 15:
                left_x = i
                break

        # Find right corner: scan right, find brightness jump
        right_x = px
        for i in range(px, len(row) - 1):
            if row[i] > row[i+1] + 15:
                right_x = i
                break

        return (
            (offset_x + left_x, pupil[1]),
            (offset_x + right_x, pupil[1])
        )

    def _detect_nose(
        self,
        img_gray: ImageArray,
        face_rect: FaceRect,
    ) -> tuple[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        dict,
    ]:
        """Detect nose base and sides using brightness profile.

        Parameters
        ----------
        img_gray : ImageArray
            Full grayscale input image.
        face_rect : FaceRect
            Face bounding box ``(x, y, w, h)``.

        Returns
        -------
        tuple
            First element is a tuple of 3 (int, int) nose points.
            Second element is a dict with diagnostic step data.
        """
        x, y, w, h = face_rect

        # Nose ROI: middle 30 % of face height, centred horizontally
        y_start = y + int(h * 0.35)
        y_end = y + int(h * 0.75)
        cx = x + w // 2

        roi = img_gray[y_start:y_end, x:x + w]
        if roi.size == 0:
            nose_y = y + int(h * 0.60)
            fallback = (
                (cx - int(w * 0.08), nose_y),
                (cx, nose_y),
                (cx + int(w * 0.08), nose_y),
            )
            return fallback, {"fallback": True, "reason": "empty_roi"}

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

        step_data = {
            "roi": roi,
            "profile": profile,
            "nose_base_y": nose_base_y,
            "row": row,
            "left_valley": left_valley if len(left_half) > 0 else None,
            "right_valley": right_valley if len(right_half) > 0 else None,
            "col_idx": col_idx,
        }
        return (
            (nose_left_x, nose_base_y),
            (cx, nose_base_y),
            (nose_right_x, nose_base_y),
        ), step_data

    def _detect_mouth(
        self,
        img_gray: ImageArray,
        face_rect: FaceRect,
    ) -> tuple[tuple[tuple[int, int], tuple[int, int]], dict]:
        """Detect mouth corners using horizontal gradient.

        Parameters
        ----------
        img_gray : ImageArray
            Full grayscale input image.
        face_rect : FaceRect
            Face bounding box ``(x, y, w, h)``.

        Returns
        -------
        tuple
            First element is a tuple of 2 (int, int) mouth corners.
            Second element is a dict with diagnostic step data.
        """
        x, y, w, h = face_rect

        # Mouth ROI: 65-85% of face height (tighter than lower third)
        y_start = y + int(h * 0.65)
        y_end = y + int(h * 0.85)
        roi = img_gray[y_start:y_end, x:x + w]

        if roi.size == 0:
            mouth_y = y + int(h * 0.75)  # Center of 65-85% range
            fallback = (x + int(w * 0.25), mouth_y), (x + int(w * 0.75), mouth_y)
            return fallback, {"fallback": True, "reason": "empty_roi"}

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

        step_data = {
            "roi": roi,
            "sobel_x": sobel_x,
            "gradient_profile": gradient_profile,
            "left_idx": left_idx,
            "right_idx": right_idx,
            "mid": mid,
        }
        return ((x + left_idx, mouth_y), (x + right_idx, mouth_y)), step_data

    def _detect_eyebrows(
        self,
        img_gray: ImageArray,
        face_rect: FaceRect,
    ) -> tuple[tuple[tuple[int, int], tuple[int, int]], dict]:
        """Detect eyebrow peaks using horizontal edge detection.

        Parameters
        ----------
        img_gray : ImageArray
            Full grayscale input image.
        face_rect : FaceRect
            Face bounding box ``(x, y, w, h)``.

        Returns
        -------
        tuple
            First element is a tuple of 2 (int, int) eyebrow peaks.
            Second element is a dict with diagnostic step data.
        """
        x, y, w, h = face_rect

        # Eyebrow ROIs: roughly top 15-40 % of face
        y_start = y + int(h * 0.10)
        y_end = y + int(h * 0.40)
        roi = img_gray[y_start:y_end, x:x + w]

        if roi.size == 0:
            brow_y = y + int(h * 0.20)
            fallback = (x + int(w * 0.30), brow_y), (x + int(w * 0.70), brow_y)
            return fallback, {"fallback": True, "reason": "empty_roi"}

        # Horizontal edges (Sobel Y)
        sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

        # Split into left/right halves
        mid_x = w // 2
        left_sobel = sobel_y[:, :mid_x]
        right_sobel = sobel_y[:, mid_x:]

        left_peak = self._find_brow_peak(left_sobel, x, y_start)
        right_peak = self._find_brow_peak(right_sobel, x + mid_x, y_start)

        step_data = {
            "roi": roi,
            "sobel_y": sobel_y,
            "left_peak": left_peak,
            "right_peak": right_peak,
        }
        return (left_peak, right_peak), step_data

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
