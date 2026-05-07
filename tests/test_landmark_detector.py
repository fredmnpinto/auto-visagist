"""Tests for the LandmarkDetector module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from visagism.landmark_detector import LandmarkDetector
from visagism.types import FaceRect, FacialLandmarks


class TestLandmarkDetector:
    """Test suite for LandmarkDetector."""

    @patch("visagism.landmark_detector.dlib.shape_predictor")
    def test_init(self, mock_shape_predictor: MagicMock, fake_model_path: Path) -> None:
        """Test that LandmarkDetector initialises with a model path."""
        detector = LandmarkDetector(fake_model_path)
        mock_shape_predictor.assert_called_once_with(str(fake_model_path))
        assert detector._predictor is not None

    @patch("visagism.landmark_detector.dlib.shape_predictor")
    def test_detect_returns_facial_landmarks(
        self, mock_shape_predictor: MagicMock, fake_model_path: Path
    ) -> None:
        """Test that detect returns a FacialLandmarks instance."""
        # Set up mock predictor
        mock_predictor = MagicMock()
        mock_shape = MagicMock()
        mock_shape.part = MagicMock(
            side_effect=lambda i: type(
                "point", (), {"x": 100 + i, "y": 100 + i // 2}
            )()
        )
        mock_predictor.return_value = mock_shape
        mock_shape_predictor.return_value = mock_predictor

        detector = LandmarkDetector(fake_model_path)

        gray = np.zeros((400, 400), dtype=np.uint8)
        face_rect: FaceRect = (50, 50, 200, 250)

        result = detector.detect(gray, face_rect, Path("/fake/test.jpg"))
        assert isinstance(result, FacialLandmarks)
        assert len(result.landmarks_68) == 68
        assert result.face_rect == face_rect
        assert result.image_path == Path("/fake/test.jpg")

    @patch("visagism.landmark_detector.dlib.shape_predictor")
    def test_detect_68_landmarks(
        self, mock_shape_predictor: MagicMock, fake_model_path: Path
    ) -> None:
        """Test that exactly 68 landmarks are returned."""
        mock_predictor = MagicMock()
        mock_shape = MagicMock()
        mock_shape.part = MagicMock(
            side_effect=lambda i: type(
                "point", (), {"x": 100 + i, "y": 100 + i // 2}
            )()
        )
        mock_predictor.return_value = mock_shape
        mock_shape_predictor.return_value = mock_predictor

        detector = LandmarkDetector(fake_model_path)

        gray = np.zeros((400, 400), dtype=np.uint8)
        result = detector.detect(
            gray, (50, 50, 200, 250), Path("/fake/test.jpg")
        )
        assert len(result.landmarks_68) == 68

    @patch("visagism.landmark_detector.dlib.shape_predictor")
    def test_detect_regions(
        self, mock_shape_predictor: MagicMock, fake_model_path: Path
    ) -> None:
        """Test that landmarks are correctly grouped by region."""
        mock_predictor = MagicMock()
        mock_shape = MagicMock()
        mock_shape.part = MagicMock(
            side_effect=lambda i: type(
                "point", (), {"x": 100 + i, "y": 100 + i // 2}
            )()
        )
        mock_predictor.return_value = mock_shape
        mock_shape_predictor.return_value = mock_predictor

        detector = LandmarkDetector(fake_model_path)

        gray = np.zeros((400, 400), dtype=np.uint8)
        result = detector.detect(
            gray, (50, 50, 200, 250), Path("/fake/test.jpg")
        )

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

        # Check region sizes
        assert len(regions["jaw"]) == 17
        assert len(regions["left_eyebrow"]) == 5
        assert len(regions["right_eyebrow"]) == 5
        assert len(regions["nose_bridge"]) == 4
        assert len(regions["nose_tip"]) == 5
        assert len(regions["left_eye"]) == 6
        assert len(regions["right_eye"]) == 6
        assert len(regions["outer_mouth"]) == 12
        assert len(regions["inner_mouth"]) == 8

    @patch("visagism.landmark_detector.dlib.shape_predictor")
    def test_detect_correct_index_mapping(
        self, mock_shape_predictor: MagicMock, fake_model_path: Path
    ) -> None:
        """Test that landmark indices map to correct regions."""
        from visagism.constants import REGION_INDICES

        mock_predictor = MagicMock()
        mock_shape = MagicMock()
        mock_shape.part = MagicMock(
            side_effect=lambda i: type(
                "point", (), {"x": i, "y": i}
            )()
        )
        mock_predictor.return_value = mock_shape
        mock_shape_predictor.return_value = mock_predictor

        detector = LandmarkDetector(fake_model_path)

        gray = np.zeros((400, 400), dtype=np.uint8)
        result = detector.detect(
            gray, (50, 50, 200, 250), Path("/fake/test.jpg")
        )

        # Verify that region points match the right indices from the full list
        for region_name, indices in REGION_INDICES.items():
            for point, expected_idx in zip(
                result.landmarks_by_region[region_name], indices
            ):
                expected_point = result.landmarks_68[expected_idx]
                assert point == expected_point

    def test_detect_group_by_region_static(self) -> None:
        """Test the static _group_by_region method."""
        landmarks_68 = [(i, i) for i in range(68)]
        from visagism.constants import REGION_INDICES

        result = LandmarkDetector._group_by_region(landmarks_68)
        assert isinstance(result, dict)
        assert len(result) == len(REGION_INDICES)
        for name, indices in REGION_INDICES.items():
            assert len(result[name]) == len(indices)

    def test_check_pose_frontal(self) -> None:
        """Test that a symmetric landmark set is classified as frontal."""
        landmarks_68 = [(0, 0)] * 68
        # Nose tip at (100, 100)
        landmarks_68[30] = (100, 100)
        # Left eye outer corner at (80, 90)
        landmarks_68[36] = (80, 90)
        # Right eye outer corner at (120, 90)
        landmarks_68[45] = (120, 90)
        # Symmetric -> frontal
        assert LandmarkDetector.check_pose(landmarks_68) is True

    def test_check_pose_frontal_tolerance(self) -> None:
        """Test that slight asymmetry (ratio ~0.9) is still frontal."""
        landmarks_68 = [(0, 0)] * 68
        landmarks_68[30] = (100, 100)
        landmarks_68[36] = (75, 90)   # slightly closer
        landmarks_68[45] = (125, 90)
        # ratio = 25 / 25 = 1.0, fine
        assert LandmarkDetector.check_pose(landmarks_68) is True

    def test_check_pose_non_frontal(self) -> None:
        """Test that a clearly asymmetric landmark set is non-frontal."""
        landmarks_68 = [(0, 0)] * 68
        landmarks_68[30] = (100, 100)
        landmarks_68[36] = (50, 90)   # much closer to nose
        landmarks_68[45] = (150, 90)  # much farther from nose
        # dist_left = sqrt(50^2 + 10^2) = sqrt(2600) ≈ 50.99
        # dist_right = sqrt(50^2 + 10^2) = sqrt(2600) ≈ 50.99
        # ratio = 1.0 -> still frontal (symmetric)

        # Let's make a clearly non-frontal case
        landmarks_68 = [(0, 0)] * 68
        landmarks_68[30] = (100, 100)
        landmarks_68[36] = (130, 95)  # far to the right of nose
        landmarks_68[45] = (105, 95)  # very close to nose
        # dist_left = sqrt(30^2 + 5^2) = sqrt(925) ≈ 30.41
        # dist_right = sqrt(5^2 + 5^2) = sqrt(50) ≈ 7.07
        # ratio = 30.41 / 7.07 ≈ 4.3 > 1.3 -> non-frontal
        assert LandmarkDetector.check_pose(landmarks_68) is False

    def test_check_pose_non_frontal_ratio_below_threshold(self) -> None:
        """Test that ratio < 0.7 is classified as non-frontal."""
        landmarks_68 = [(0, 0)] * 68
        landmarks_68[30] = (100, 100)
        landmarks_68[36] = (105, 95)  # very close to nose
        landmarks_68[45] = (130, 95)  # far to the right
        # dist_left = sqrt(5^2 + 5^2) = sqrt(50) ≈ 7.07
        # dist_right = sqrt(30^2 + 5^2) = sqrt(925) ≈ 30.41
        # ratio = 7.07 / 30.41 ≈ 0.23 < 0.7 -> non-frontal
        assert LandmarkDetector.check_pose(landmarks_68) is False

    def test_check_pose_frontal_boundary(self) -> None:
        """Test that ratio exactly 0.7 is frontal (boundary inclusive)."""
        landmarks_68 = [(0, 0)] * 68
        # Place landmarks so ratio = 0.7 exactly
        landmarks_68[30] = (0, 0)
        landmarks_68[36] = (7, 0)    # dist = 7
        landmarks_68[45] = (10, 0)   # dist = 10
        # ratio = 0.7 -> should be frontal (boundary inclusive)
        assert LandmarkDetector.check_pose(landmarks_68) is True

    def test_check_pose_non_frontal_boundary(self) -> None:
        """Test that ratio 0.69 is non-frontal (just below threshold)."""
        landmarks_68 = [(0, 0)] * 68
        landmarks_68[30] = (0, 0)
        landmarks_68[36] = (69, 0)   # dist = 69
        landmarks_68[45] = (100, 0)  # dist = 100
        # ratio = 0.69 < 0.7 -> non-frontal
        assert LandmarkDetector.check_pose(landmarks_68) is False

    def test_check_pose_upper_boundary(self) -> None:
        """Test that ratio exactly 1.3 is frontal (boundary inclusive)."""
        landmarks_68 = [(0, 0)] * 68
        landmarks_68[30] = (0, 0)
        landmarks_68[36] = (130, 0)  # dist = 130
        landmarks_68[45] = (100, 0)  # dist = 100
        # ratio = 1.3 -> frontal (boundary inclusive)
        assert LandmarkDetector.check_pose(landmarks_68) is True

    def test_check_pose_upper_boundary_exceeded(self) -> None:
        """Test that ratio 1.31 is non-frontal (just above threshold)."""
        landmarks_68 = [(0, 0)] * 68
        landmarks_68[30] = (0, 0)
        landmarks_68[36] = (131, 0)  # dist = 131
        landmarks_68[45] = (100, 0)  # dist = 100
        # ratio = 1.31 > 1.3 -> non-frontal
        assert LandmarkDetector.check_pose(landmarks_68) is False

    def test_check_pose_degenerate(self) -> None:
        """Test degenerate case (zero distance) returns True."""
        landmarks_68 = [(0, 0)] * 68
        # All points at origin, dist_right = 0
        assert LandmarkDetector.check_pose(landmarks_68) is True
