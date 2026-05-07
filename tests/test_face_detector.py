"""Tests for the FaceDetector module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from visagism.errors import NoFaceDetectedError


class TestFaceDetector:
    """Test suite for FaceDetector."""

    def test_init_success(self) -> None:
        """Test that FaceDetector initialises successfully."""
        from visagism.face_detector import FaceDetector

        detector = FaceDetector()
        assert detector._cascade is not None

    def test_detect_no_face(self) -> None:
        """Test that NoFaceDetectedError is raised when no face found."""
        from visagism.face_detector import FaceDetector

        detector = FaceDetector()
        blank = np.zeros((400, 400), dtype=np.uint8)
        with pytest.raises(NoFaceDetectedError, match="No face detected"):
            detector.detect(blank, (400, 400))

    @patch("visagism.face_detector._locate_haar_cascade", return_value="/fake/path.xml")
    def test_detect_returns_face_rect(self, mock_locate: MagicMock) -> None:
        """Test that detect returns a valid FaceRect tuple (with mock)."""
        # We need to create a mock cascade where detectMultiScale returns fake faces
        from visagism.face_detector import FaceDetector, cv2

        # Create mock cascade
        mock_cascade = MagicMock(spec=cv2.CascadeClassifier)
        fake_faces = np.array([[50, 50, 200, 250]], dtype=np.int32)
        mock_cascade.detectMultiScale.return_value = fake_faces

        detector = FaceDetector()
        detector._cascade = mock_cascade

        gray = np.zeros((400, 400), dtype=np.uint8)
        result = detector.detect(gray, (400, 400))
        assert len(result) == 4
        x, y, w, h = result
        assert x >= 0
        assert y >= 0
        assert w > 0
        assert h > 0

    @patch("visagism.face_detector._locate_haar_cascade", return_value="/fake/path.xml")
    def test_detect_single_face(self, mock_locate: MagicMock) -> None:
        """Test detection with mocked cascade returning one face."""
        from visagism.face_detector import FaceDetector, cv2

        mock_cascade = MagicMock(spec=cv2.CascadeClassifier)
        fake_faces = np.array([[50, 50, 200, 250]], dtype=np.int32)
        mock_cascade.detectMultiScale.return_value = fake_faces

        detector = FaceDetector()
        detector._cascade = mock_cascade

        gray = np.zeros((400, 400), dtype=np.uint8)
        result = detector.detect(gray, (400, 400))
        assert len(result) == 4

    @patch("visagism.face_detector._locate_haar_cascade", return_value="/fake/path.xml")
    def test_detect_largest_face_selected(self, mock_locate: MagicMock) -> None:
        """Test that when multiple faces are detected, the largest is used."""
        from visagism.face_detector import FaceDetector, cv2

        mock_cascade = MagicMock(spec=cv2.CascadeClassifier)
        fake_faces = np.array([
            [10, 10, 50, 50],     # small face: 2500 area
            [30, 30, 200, 250],    # large face: 50000 area
        ], dtype=np.int32)
        mock_cascade.detectMultiScale.return_value = fake_faces

        detector = FaceDetector()
        detector._cascade = mock_cascade

        gray = np.zeros((400, 400), dtype=np.uint8)
        result = detector.detect(gray, (400, 400))
        # Large face at (30,30): 200x250 -> expand 20% -> dx=20, dy=25
        # x_new = max(0, 30-20)=10, y_new = max(0, 30-25)=5
        # w_new = min(400-10, 200+2*20)=240, h_new = min(400-5, 250+2*25)=300
        assert result == (10, 5, 240, 300)

    @patch("visagism.face_detector._locate_haar_cascade", return_value="/fake/path.xml")
    def test_detect_bbox_expansion(self, mock_locate: MagicMock) -> None:
        """Test that bounding box is expanded by 20%."""
        from visagism.face_detector import FaceDetector, cv2

        mock_cascade = MagicMock(spec=cv2.CascadeClassifier)
        fake_faces = np.array([[100, 100, 100, 100]], dtype=np.int32)
        mock_cascade.detectMultiScale.return_value = fake_faces

        detector = FaceDetector()
        detector._cascade = mock_cascade

        gray = np.zeros((400, 400), dtype=np.uint8)
        result = detector.detect(gray, (400, 400))
        # Original: x=100, y=100, w=100, h=100
        # Expand by 20%: dx=10, dy=10
        # x_new=90, y_new=90, w_new=120, h_new=120
        assert result == (90, 90, 120, 120)

    @patch("visagism.face_detector._locate_haar_cascade", return_value="/fake/path.xml")
    def test_detect_bbox_clamped_to_image(self, mock_locate: MagicMock) -> None:
        """Test that expanded bbox does not exceed image bounds."""
        from visagism.face_detector import FaceDetector, cv2

        mock_cascade = MagicMock(spec=cv2.CascadeClassifier)
        fake_faces = np.array([[0, 0, 50, 50]], dtype=np.int32)
        mock_cascade.detectMultiScale.return_value = fake_faces

        detector = FaceDetector()
        detector._cascade = mock_cascade

        gray = np.zeros((100, 100), dtype=np.uint8)
        result = detector.detect(gray, (100, 100))
        x, y, w, h = result
        assert x >= 0
        assert y >= 0
        assert x + w <= 100
        assert y + h <= 100

    @patch("visagism.face_detector._locate_haar_cascade", return_value="/fake/path.xml")
    def test_detect_small_face_warning(
        self, mock_locate: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that a warning is printed for small faces."""
        from visagism.face_detector import FaceDetector, cv2

        mock_cascade = MagicMock(spec=cv2.CascadeClassifier)
        # Face area = 30*30 = 900, image area = 400*400 = 160000
        # 900 / 160000 = 0.0056 < 0.10 threshold
        fake_faces = np.array([[100, 100, 30, 30]], dtype=np.int32)
        mock_cascade.detectMultiScale.return_value = fake_faces

        detector = FaceDetector()
        detector._cascade = mock_cascade

        gray = np.zeros((400, 400), dtype=np.uint8)
        detector.detect(gray, (400, 400))
        captured = capsys.readouterr()
        assert "small" in captured.out.lower()

    @patch("visagism.face_detector._locate_haar_cascade", return_value="/fake/path.xml")
    def test_detect_multiple_faces_warning(
        self, mock_locate: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that a message is printed when multiple faces detected."""
        from visagism.face_detector import FaceDetector, cv2

        mock_cascade = MagicMock(spec=cv2.CascadeClassifier)
        fake_faces = np.array([
            [10, 10, 100, 100],
            [200, 200, 150, 150],
        ], dtype=np.int32)
        mock_cascade.detectMultiScale.return_value = fake_faces

        detector = FaceDetector()
        detector._cascade = mock_cascade

        gray = np.zeros((400, 400), dtype=np.uint8)
        detector.detect(gray, (400, 400))
        captured = capsys.readouterr()
        assert "Multiple faces" in captured.out
