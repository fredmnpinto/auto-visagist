"""Tests for the HairlineDetector module."""

from __future__ import annotations

import numpy as np
import pytest

from visagism.hairline_detector import HairlineDetector
from visagism.types import FacialLandmarks, LandmarkRegions


class TestHairlineDetector:
    """Test suite for HairlineDetector."""

    def test_detect_avg_eyebrow_y(self, sample_landmarks: FacialLandmarks) -> None:
        """Test that average eyebrow y is computed correctly."""
        detector = HairlineDetector()

        # Compute expected average eyebrow y manually
        left_brow = sample_landmarks.landmarks_by_region["left_eyebrow"]
        right_brow = sample_landmarks.landmarks_by_region["right_eyebrow"]
        all_brow_pts = left_brow + right_brow
        expected_avg = int(np.mean([pt[1] for pt in all_brow_pts]))

        # Create a blank image large enough
        img_gray = np.zeros((400, 400), dtype=np.uint8)
        img_gray[:] = 128

        # With no edges, it will fallback, but we can verify the avg is used
        # by checking the fallback result matches our calculation
        nose_base_y = sample_landmarks.landmarks_68[33][1]
        medium_third = nose_base_y - expected_avg
        expected_fallback = max(
            expected_avg - medium_third, sample_landmarks.face_rect[1]
        )

        with pytest.warns(UserWarning, match="No strong hairline edge detected"):
            result = detector.detect(img_gray, sample_landmarks)

        assert result["hairline_y"] == expected_fallback
        assert result["method"] == "fallback"

    def test_detect_fallback_no_edge(self, sample_landmarks: FacialLandmarks) -> None:
        """Test fallback when no strong edge is found in the forehead region."""
        detector = HairlineDetector()
        img_gray = np.zeros((400, 400), dtype=np.uint8)
        img_gray[:] = 128  # uniform gray — no edges

        with pytest.warns(UserWarning, match="No strong hairline edge detected"):
            result = detector.detect(img_gray, sample_landmarks)

        # Fallback should be clamped to face_rect[1] at minimum
        assert result["hairline_y"] >= sample_landmarks.face_rect[1]
        assert result["method"] == "fallback"

    def test_detect_fallback_calculation(self) -> None:
        """Test the fallback geometric calculation."""
        detector = HairlineDetector()

        # Construct landmarks with known values
        # eyebrow_y = 150, nose_base_y (point 33) = 200
        # medium_third = 200 - 150 = 50
        # hairline_y = 150 - 50 = 100
        # face_rect[1] = 50 -> max(100, 50) = 100
        from pathlib import Path
        from visagism.constants import REGION_INDICES

        landmarks_68 = [(0, 0)] * 68
        landmarks_68[33] = (150, 200)  # nose tip y = 200

        # Set eyebrow points to y = 150
        for idx in REGION_INDICES["left_eyebrow"] + REGION_INDICES["right_eyebrow"]:
            landmarks_68[idx] = (100, 150)

        landmarks_by_region: LandmarkRegions = {
            name: [landmarks_68[i] for i in indices]
            for name, indices in REGION_INDICES.items()
        }

        landmarks = FacialLandmarks(
            image_path=Path("/fake/test.jpg"),
            face_rect=(50, 50, 200, 250),
            landmarks_68=landmarks_68,
            landmarks_by_region=landmarks_by_region,
        )

        img_gray = np.zeros((400, 400), dtype=np.uint8)
        img_gray[:] = 128

        with pytest.warns(UserWarning, match="No strong hairline edge detected"):
            result = detector.detect(img_gray, landmarks)

        assert result["hairline_y"] == 100
        assert result["method"] == "fallback"

    def test_detect_fallback_clamped(self) -> None:
        """Test that fallback is clamped to face_rect top when calculated above."""
        detector = HairlineDetector()

        from pathlib import Path
        from visagism.constants import REGION_INDICES

        landmarks_68 = [(0, 0)] * 68
        # nose_base_y = 120, eyebrow_y = 100
        # medium_third = 20, hairline_y = 80
        # face_rect[1] = 90 -> max(80, 90) = 90
        landmarks_68[33] = (150, 120)

        for idx in REGION_INDICES["left_eyebrow"] + REGION_INDICES["right_eyebrow"]:
            landmarks_68[idx] = (100, 100)

        landmarks_by_region: LandmarkRegions = {
            name: [landmarks_68[i] for i in indices]
            for name, indices in REGION_INDICES.items()
        }

        landmarks = FacialLandmarks(
            image_path=Path("/fake/test.jpg"),
            face_rect=(50, 90, 200, 250),
            landmarks_68=landmarks_68,
            landmarks_by_region=landmarks_by_region,
        )

        img_gray = np.zeros((400, 400), dtype=np.uint8)
        img_gray[:] = 128

        with pytest.warns(UserWarning, match="No strong hairline edge detected"):
            result = detector.detect(img_gray, landmarks)

        assert result["hairline_y"] == 90
        assert result["method"] == "fallback"

    def test_detect_invalid_roi(self) -> None:
        """Test fallback when the search region is invalid."""
        detector = HairlineDetector()

        from pathlib import Path
        from visagism.constants import REGION_INDICES

        landmarks_68 = [(0, 0)] * 68
        landmarks_68[33] = (150, 200)

        # Set eyebrows very high so avg_eyebrow_y <= face_rect[1]
        for idx in REGION_INDICES["left_eyebrow"] + REGION_INDICES["right_eyebrow"]:
            landmarks_68[idx] = (100, 40)

        landmarks_by_region: LandmarkRegions = {
            name: [landmarks_68[i] for i in indices]
            for name, indices in REGION_INDICES.items()
        }

        landmarks = FacialLandmarks(
            image_path=Path("/fake/test.jpg"),
            face_rect=(50, 50, 200, 250),
            landmarks_68=landmarks_68,
            landmarks_by_region=landmarks_by_region,
        )

        img_gray = np.zeros((400, 400), dtype=np.uint8)
        img_gray[:] = 128

        # Invalid ROI (y_start=50, y_end=40) triggers fallback without warning
        result = detector.detect(img_gray, landmarks)
        assert result["hairline_y"] >= landmarks.face_rect[1]
        assert result["method"] == "fallback"

    def test_detect_returns_expected_keys(
        self, sample_landmarks, sample_image_gray
    ):
        """Test that detect returns all expected keys."""
        detector = HairlineDetector()
        steps = detector.detect(sample_image_gray, sample_landmarks)

        expected_keys = {
            "roi_raw", "roi_enhanced", "row_intensities", "gradient",
            "abs_gradient", "max_gradient_idx", "max_gradient_value",
            "max_gradient_value_full", "median_gradient", "gradient_ratio",
            "hairline_y", "method", "roi_coords", "avg_eyebrow_y",
            "face_rect", "searchable_rows"
        }
        assert set(steps.keys()) == expected_keys

    def test_detect_with_strong_gradient(self) -> None:
        """Test detection when a strong vertical intensity gradient is present."""
        detector = HairlineDetector()

        from pathlib import Path
        from visagism.constants import REGION_INDICES

        # face_rect = (50, 50, 200, 250)
        # avg_eyebrow_y should be > the gradient row
        # ROI is y=[50, avg_eyebrow_y], so gradient at local row 30 -> global y=80

        landmarks_68 = [(0, 0)] * 68
        landmarks_68[33] = (150, 200)

        # Eyebrows at y = 150 -> avg_eyebrow_y = 150
        for idx in REGION_INDICES["left_eyebrow"] + REGION_INDICES["right_eyebrow"]:
            landmarks_68[idx] = (100, 150)

        landmarks_by_region: LandmarkRegions = {
            name: [landmarks_68[i] for i in indices]
            for name, indices in REGION_INDICES.items()
        }

        landmarks = FacialLandmarks(
            image_path=Path("/fake/test.jpg"),
            face_rect=(50, 50, 200, 250),
            landmarks_68=landmarks_68,
            landmarks_by_region=landmarks_by_region,
        )

        # Create image with a strong intensity transition at y=80
        # Top half of ROI (y=50-80) is dark (hair),
        # bottom half (y=80-150) is bright (skin)
        img_gray = np.zeros((400, 400), dtype=np.uint8)
        img_gray[:] = 50
        img_gray[50:81, 50:250] = 50   # dark region (simulating hair)
        img_gray[81:150, 50:250] = 200  # bright region (simulating skin)

        result = detector.detect(img_gray, landmarks)

        # The gradient should be detected near the transition at y=80
        assert abs(result["hairline_y"] - 80) <= 2
        assert result["method"] == "edge"

    # ------------------------------------------------------------------
    # Dict key consistency and backward-compatibility (refactoring tests)
    # ------------------------------------------------------------------

    EXPECTED_KEYS = {
        "roi_raw", "roi_enhanced", "row_intensities", "gradient",
        "abs_gradient", "max_gradient_idx", "max_gradient_value",
        "max_gradient_value_full", "median_gradient", "gradient_ratio",
        "hairline_y", "method", "roi_coords", "avg_eyebrow_y",
        "face_rect", "searchable_rows",
    }

    def test_detect_returns_exactly_16_keys_fallback(self, sample_landmarks) -> None:
        """Fallback path must return all 16 keys."""
        detector = HairlineDetector()
        img_gray = np.full((400, 400), 128, dtype=np.uint8)

        with pytest.warns(UserWarning, match="No strong hairline edge detected"):
            result = detector.detect(img_gray, sample_landmarks)

        assert set(result.keys()) == self.EXPECTED_KEYS
        assert len(result) == 16

    def test_detect_returns_exactly_16_keys_invalid_roi(self) -> None:
        """Invalid ROI path must return all 16 keys."""
        detector = HairlineDetector()
        from pathlib import Path
        from visagism.constants import REGION_INDICES

        landmarks_68 = [(0, 0)] * 68
        landmarks_68[33] = (150, 200)
        for idx in REGION_INDICES["left_eyebrow"] + REGION_INDICES["right_eyebrow"]:
            landmarks_68[idx] = (100, 40)

        landmarks_by_region: LandmarkRegions = {
            name: [landmarks_68[i] for i in indices]
            for name, indices in REGION_INDICES.items()
        }
        landmarks = FacialLandmarks(
            image_path=Path("/fake/test.jpg"),
            face_rect=(50, 50, 200, 250),
            landmarks_68=landmarks_68,
            landmarks_by_region=landmarks_by_region,
        )
        img_gray = np.full((400, 400), 128, dtype=np.uint8)

        result = detector.detect(img_gray, landmarks)
        assert set(result.keys()) == self.EXPECTED_KEYS
        assert len(result) == 16

    def test_detect_returns_exactly_16_keys_edge(self) -> None:
        """Edge-detection path must return all 16 keys."""
        detector = HairlineDetector()
        from pathlib import Path
        from visagism.constants import REGION_INDICES

        landmarks_68 = [(0, 0)] * 68
        landmarks_68[33] = (150, 200)
        for idx in REGION_INDICES["left_eyebrow"] + REGION_INDICES["right_eyebrow"]:
            landmarks_68[idx] = (100, 150)

        landmarks_by_region: LandmarkRegions = {
            name: [landmarks_68[i] for i in indices]
            for name, indices in REGION_INDICES.items()
        }
        landmarks = FacialLandmarks(
            image_path=Path("/fake/test.jpg"),
            face_rect=(50, 50, 200, 250),
            landmarks_68=landmarks_68,
            landmarks_by_region=landmarks_by_region,
        )

        img_gray = np.zeros((400, 400), dtype=np.uint8)
        img_gray[:] = 50
        img_gray[50:81, 50:250] = 50
        img_gray[81:150, 50:250] = 200

        result = detector.detect(img_gray, landmarks)
        assert set(result.keys()) == self.EXPECTED_KEYS
        assert len(result) == 16

    def test_fallback_dict_includes_max_gradient_value_full_and_searchable_rows(
        self, sample_landmarks
    ) -> None:
        """Fallback responses must contain the two keys added during refactoring."""
        detector = HairlineDetector()
        img_gray = np.full((400, 400), 128, dtype=np.uint8)

        with pytest.warns(UserWarning, match="No strong hairline edge detected"):
            result = detector.detect(img_gray, sample_landmarks)

        assert "max_gradient_value_full" in result
        assert "searchable_rows" in result
        assert isinstance(result["max_gradient_value_full"], float)
        assert isinstance(result["searchable_rows"], int)
