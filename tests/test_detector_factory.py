"""Tests for the detector_factory module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from visagism.detector_factory import SUPPORTED_DETECTORS, create_landmark_detector
from visagism.errors import DetectionError
from visagism.landmark_detector import LandmarkDetector
from visagism.nonml_landmark_detector import NonMLLandmarkDetector


class TestDetectorFactory:
    """Test suite for create_landmark_detector."""

    @patch("visagism.landmark_detector.dlib.shape_predictor")
    def test_factory_returns_landmark_detector_for_dlib(
        self, mock_shape_predictor: MagicMock, fake_model_path: Path
    ) -> None:
        """Factory returns LandmarkDetector when type is 'dlib'."""
        detector = create_landmark_detector("dlib", fake_model_path)
        assert isinstance(detector, LandmarkDetector)

    def test_factory_returns_nonml_detector_for_nonml(
        self, fake_model_path: Path
    ) -> None:
        """Factory returns NonMLLandmarkDetector when type is 'nonml'."""
        detector = create_landmark_detector("nonml", fake_model_path)
        assert isinstance(detector, NonMLLandmarkDetector)

    def test_factory_raises_for_invalid_type(self, fake_model_path: Path) -> None:
        """Factory raises DetectionError for unsupported detector type."""
        with pytest.raises(DetectionError, match="Unsupported detector type"):
            create_landmark_detector("invalid", fake_model_path)

    def test_supported_detectors_set(self) -> None:
        """SUPPORTED_DETECTORS contains expected values."""
        assert SUPPORTED_DETECTORS == {"dlib", "nonml"}
