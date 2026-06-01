"""Tests for the NonMLLandmarkDetector module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from visagism.nonml_landmark_detector import NonMLLandmarkDetector
from visagism.types import FaceRect, FacialLandmarks


class TestNonMLLandmarkDetector:
    """Test suite for NonMLLandmarkDetector."""

    def test_init_accepts_model_path(self, fake_model_path: Path) -> None:
        """Test that __init__ accepts a model path (ignored)."""
        detector = NonMLLandmarkDetector(fake_model_path)
        assert detector._model_path == fake_model_path

    def test_detect_returns_facial_landmarks(
        self, fake_model_path: Path
    ) -> None:
        """Test that detect returns a FacialLandmarks instance."""
        detector = NonMLLandmarkDetector(fake_model_path)

        # Create a synthetic face-like image
        img_gray = np.full((400, 400), 180, dtype=np.uint8)
        # Dark eye regions
        cv2.ellipse(img_gray, (120, 150), (20, 10), 0, 0, 360, 50, -1)
        cv2.ellipse(img_gray, (280, 150), (20, 10), 0, 0, 360, 50, -1)
        # Dark nostrils
        cv2.circle(img_gray, (170, 240), 5, 60, -1)
        cv2.circle(img_gray, (230, 240), 5, 60, -1)
        # Mouth line
        cv2.line(img_gray, (150, 320), (250, 320), 40, 3)
        # Eyebrow edges
        cv2.line(img_gray, (110, 110), (150, 105), 80, 2)
        cv2.line(img_gray, (250, 105), (290, 110), 80, 2)

        face_rect: FaceRect = (50, 50, 300, 350)
        result = detector.detect(img_gray, face_rect, Path("/fake/test.jpg"))

        assert isinstance(result, FacialLandmarks)
        assert len(result.landmarks_68) == 68
        assert result.face_rect == face_rect
        assert result.image_path == Path("/fake/test.jpg")

    def test_detect_produces_68_landmarks(self, fake_model_path: Path) -> None:
        """Test that exactly 68 landmarks are returned."""
        detector = NonMLLandmarkDetector(fake_model_path)
        img_gray = np.full((400, 400), 180, dtype=np.uint8)
        face_rect: FaceRect = (50, 50, 300, 350)

        result = detector.detect(img_gray, face_rect, Path("/fake/test.jpg"))
        assert len(result.landmarks_68) == 68

    def test_detect_14_anchor_points_present(self, fake_model_path: Path) -> None:
        """Test that the 14 anchor points are not (-1, -1)."""
        detector = NonMLLandmarkDetector(fake_model_path)

        img_gray = np.full((400, 400), 180, dtype=np.uint8)
        cv2.ellipse(img_gray, (120, 150), (20, 10), 0, 0, 360, 50, -1)
        cv2.ellipse(img_gray, (280, 150), (20, 10), 0, 0, 360, 50, -1)
        cv2.circle(img_gray, (170, 240), 5, 60, -1)
        cv2.circle(img_gray, (230, 240), 5, 60, -1)
        cv2.line(img_gray, (150, 320), (250, 320), 40, 3)
        cv2.line(img_gray, (110, 110), (150, 105), 80, 2)
        cv2.line(img_gray, (250, 105), (290, 110), 80, 2)

        face_rect: FaceRect = (50, 50, 300, 350)
        result = detector.detect(img_gray, face_rect, Path("/fake/test.jpg"))

        anchor_indices = [0, 8, 16, 31, 33, 35, 36, 39, 42, 45, 48, 54, 19, 24]
        for idx in anchor_indices:
            assert result.landmarks_68[idx] != (-1, -1), f"Anchor {idx} is missing"

    def test_detect_54_missing_points(self, fake_model_path: Path) -> None:
        """Test that the remaining 54 points are (-1, -1)."""
        detector = NonMLLandmarkDetector(fake_model_path)
        img_gray = np.full((400, 400), 180, dtype=np.uint8)
        face_rect: FaceRect = (50, 50, 300, 350)

        result = detector.detect(img_gray, face_rect, Path("/fake/test.jpg"))

        anchor_indices = {0, 8, 16, 31, 33, 35, 36, 39, 42, 45, 48, 54, 19, 24}
        missing_count = 0
        for i, pt in enumerate(result.landmarks_68):
            if i not in anchor_indices:
                assert pt == (-1, -1), f"Point {i} should be missing"
                missing_count += 1

        assert missing_count == 54

    def test_detect_groups_by_region(self, fake_model_path: Path) -> None:
        """Test that landmarks are grouped by region."""
        detector = NonMLLandmarkDetector(fake_model_path)
        img_gray = np.full((400, 400), 180, dtype=np.uint8)
        face_rect: FaceRect = (50, 50, 300, 350)

        result = detector.detect(img_gray, face_rect, Path("/fake/test.jpg"))

        regions = result.landmarks_by_region
        assert "jaw" in regions
        assert "left_eyebrow" in regions
        assert "right_eyebrow" in regions
        assert "nose_bridge" in regions
        assert "nose_tip" in regions
        assert "left_eye" in regions
        assert "right_eye" in regions
        assert "outer_mouth" in regions
        assert "inner_mouth" in regions

    def test_jawline_points_use_face_rect(self, fake_model_path: Path) -> None:
        """Test that jawline points are derived from face_rect geometry."""
        detector = NonMLLandmarkDetector(fake_model_path)
        img_gray = np.full((400, 400), 180, dtype=np.uint8)
        face_rect: FaceRect = (50, 50, 200, 250)

        result = detector.detect(img_gray, face_rect, Path("/fake/test.jpg"))

        x, y, w, h = face_rect
        assert result.landmarks_68[0] == (x, y + int(h * 0.85))
        assert result.landmarks_68[8] == (x + w // 2, y + h)
        assert result.landmarks_68[16] == (x + w, y + int(h * 0.85))

    def test_detect_with_empty_image_fallbacks(self, fake_model_path: Path) -> None:
        """Test that detection falls back gracefully on empty/invalid ROI."""
        detector = NonMLLandmarkDetector(fake_model_path)
        img_gray = np.zeros((400, 400), dtype=np.uint8)
        face_rect: FaceRect = (50, 50, 300, 350)

        result = detector.detect(img_gray, face_rect, Path("/fake/test.jpg"))

        anchor_indices = [0, 8, 16, 31, 33, 35, 36, 39, 42, 45, 48, 54, 19, 24]
        for idx in anchor_indices:
            assert result.landmarks_68[idx] != (-1, -1), f"Anchor {idx} is missing"

    def test_detect_with_zero_size_face_rect(self, fake_model_path: Path) -> None:
        """Test detection with degenerate face rect."""
        detector = NonMLLandmarkDetector(fake_model_path)
        img_gray = np.full((400, 400), 180, dtype=np.uint8)
        face_rect: FaceRect = (0, 0, 0, 0)

        result = detector.detect(img_gray, face_rect, Path("/fake/test.jpg"))
        assert len(result.landmarks_68) == 68
        # Jaw points should still be computed (all at origin)
        assert result.landmarks_68[0] == (0, 0)
        assert result.landmarks_68[8] == (0, 0)
        assert result.landmarks_68[16] == (0, 0)
