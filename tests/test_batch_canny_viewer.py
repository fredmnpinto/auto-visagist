"""Tests for the batch_canny_viewer script."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from scripts import batch_canny_viewer as bcv  # noqa: E402


class TestParseOperation:
    """Tests for _parse_operation."""

    def test_valid_operations(self) -> None:
        """All valid operation names should be accepted (case-insensitive)."""
        for op in ("close", "open", "dilate", "erode"):
            assert bcv._parse_operation(op) == op
            assert bcv._parse_operation(op.upper()) == op
            assert bcv._parse_operation(f"  {op}  ") == op

    def test_invalid_operation(self) -> None:
        """An unknown operation should raise ArgumentTypeError."""
        with pytest.raises(SystemExit):
            # argparse catches ArgumentTypeError and exits
            parser = pytest.importorskip("argparse").ArgumentParser()
            parser.add_argument("--op", type=bcv._parse_operation)
            parser.parse_args(["--op", "blur"])


class TestParseKernelSize:
    """Tests for _parse_kernel_size."""

    def test_positive_integer(self) -> None:
        """Positive integers should be returned as-is."""
        assert bcv._parse_kernel_size("5") == 5
        assert bcv._parse_kernel_size("  3  ") == 3

    def test_non_integer(self) -> None:
        """Non-integer strings should raise ArgumentTypeError."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            bcv._parse_kernel_size("abc")

    def test_zero(self) -> None:
        """Zero should raise ArgumentTypeError."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            bcv._parse_kernel_size("0")

    def test_negative(self) -> None:
        """Negative values should raise ArgumentTypeError."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            bcv._parse_kernel_size("-3")


class TestFindImages:
    """Tests for _find_images."""

    def test_missing_directory(self, tmp_path: Path) -> None:
        """A missing directory should raise FileNotFoundError."""
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            bcv._find_images(missing)

    def test_not_a_directory(self, tmp_path: Path) -> None:
        """A file path should raise NotADirectoryError."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("hello")
        with pytest.raises(NotADirectoryError):
            bcv._find_images(file_path)

    def test_finds_supported_images(self, tmp_path: Path) -> None:
        """Only supported extensions should be discovered."""
        (tmp_path / "a.jpg").write_bytes(b"fake")
        (tmp_path / "b.png").write_bytes(b"fake")
        (tmp_path / "c.gif").write_bytes(b"fake")
        (tmp_path / "d.txt").write_bytes(b"fake")

        found = bcv._find_images(tmp_path)
        names = [p.name for p in found]
        assert "a.jpg" in names
        assert "b.png" in names
        assert "c.gif" not in names
        assert "d.txt" not in names

    def test_case_insensitive_extensions(self, tmp_path: Path) -> None:
        """Upper-case extensions should also be matched."""
        (tmp_path / "a.JPG").write_bytes(b"fake")
        (tmp_path / "b.PNG").write_bytes(b"fake")

        found = bcv._find_images(tmp_path)
        names = [p.name for p in found]
        assert "a.JPG" in names
        assert "b.PNG" in names


class TestApplyMorphology:
    """Tests for _apply_morphology."""

    def test_close_operation(self) -> None:
        """Closing should not crash and return a grayscale image."""
        img = np.zeros((50, 50), dtype=np.uint8)
        img[20:30, 20:30] = 255
        result = bcv._apply_morphology(img, "close", 3)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_open_operation(self) -> None:
        """Opening should not crash and return a grayscale image."""
        img = np.ones((50, 50), dtype=np.uint8) * 255
        img[20:30, 20:30] = 0
        result = bcv._apply_morphology(img, "open", 3)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_dilate_operation(self) -> None:
        """Dilation should expand bright regions."""
        img = np.zeros((50, 50), dtype=np.uint8)
        img[24:26, 24:26] = 255
        result = bcv._apply_morphology(img, "dilate", 3)
        assert np.count_nonzero(result) > np.count_nonzero(img)

    def test_erode_operation(self) -> None:
        """Erosion should shrink bright regions."""
        img = np.zeros((50, 50), dtype=np.uint8)
        img[10:40, 10:40] = 255
        result = bcv._apply_morphology(img, "erode", 3)
        assert np.count_nonzero(result) < np.count_nonzero(img)


class TestRunCanny:
    """Tests for _run_canny."""

    def test_canny_output_shape(self) -> None:
        """Canny output should match input shape."""
        img = np.random.randint(0, 256, (60, 80), dtype=np.uint8)
        edges = bcv._run_canny(img, 30, 100)
        assert edges.shape == img.shape
        assert edges.dtype == np.uint8

    def test_canny_detects_edges(self) -> None:
        """Canny should detect edges in a simple gradient image."""
        img = np.zeros((50, 50), dtype=np.uint8)
        img[:, 25:] = 255
        edges = bcv._run_canny(img, 50, 150)
        assert np.count_nonzero(edges) > 0


class TestResizeToHeight:
    """Tests for _resize_to_height."""

    def test_same_height(self) -> None:
        """Image already at target height should be returned unchanged."""
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        result = bcv._resize_to_height(img, 100)
        assert result.shape == img.shape

    def test_resize_larger(self) -> None:
        """Resizing to a larger height should increase dimensions."""
        img = np.zeros((50, 25, 3), dtype=np.uint8)
        result = bcv._resize_to_height(img, 100)
        assert result.shape[0] == 100
        assert result.shape[1] == 50

    def test_resize_smaller(self) -> None:
        """Resizing to a smaller height should decrease dimensions."""
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        result = bcv._resize_to_height(img, 50)
        assert result.shape[0] == 50
        assert result.shape[1] == 25


class TestProcessImage:
    """Tests for _process_image."""

    def test_corrupted_image_returns_none(self, tmp_path: Path) -> None:
        """A corrupted image should return None and not crash."""
        bad_image = tmp_path / "bad.jpg"
        bad_image.write_text("not an image")
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        with patch("scripts.batch_canny_viewer.warnings.warn") as mock_warn:
            result = bcv._process_image(
                bad_image, "close", 3, 30, 100, output_dir, False
            )
        assert result is None
        mock_warn.assert_called_once()

    def test_successful_processing(self, tmp_path: Path) -> None:
        """A valid image should be saved and its path returned."""
        img_path = tmp_path / "test.png"
        # Create a simple synthetic image
        img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = bcv._process_image(
            img_path, "close", 3, 30, 100, output_dir, False
        )
        assert result is not None
        assert result.exists()
        assert result.name == "test_canny.png"

    def test_visualize_calls_opencv(self, tmp_path: Path) -> None:
        """When visualize=True, OpenCV window functions should be called."""
        img_path = tmp_path / "test.png"
        img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        with patch("scripts.batch_canny_viewer.cv2.namedWindow") as mock_named, \
             patch("scripts.batch_canny_viewer.cv2.imshow") as mock_show, \
             patch("scripts.batch_canny_viewer.cv2.waitKey",
                   return_value=0) as mock_wait, \
             patch("scripts.batch_canny_viewer.cv2.destroyWindow") as mock_destroy:
            bcv._process_image(
                img_path, "open", 3, 30, 100, output_dir, True
            )

        mock_named.assert_called_once()
        mock_show.assert_called_once()
        mock_wait.assert_called_once_with(0)
        mock_destroy.assert_called_once()

    def test_with_detectors_no_face(self, tmp_path: Path) -> None:
        """When detectors are provided but no face is found, warn and return path."""
        img_path = tmp_path / "test.png"
        img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        mock_face_detector = MagicMock()
        mock_face_detector.detect.return_value = None

        with patch("scripts.batch_canny_viewer.warnings.warn") as mock_warn:
            result = bcv._process_image(
                img_path, "close", 3, 30, 100, output_dir, False,
                face_detector=mock_face_detector,
                landmark_detector=MagicMock(),
                hairline_detector=MagicMock(),
            )

        assert result is not None
        assert result.exists()
        mock_warn.assert_called_once()
        assert "No face detected" in str(mock_warn.call_args)

    def test_with_detectors_success(self, tmp_path: Path) -> None:
        """When detectors work, hairline overlay should be included in visualization."""
        img_path = tmp_path / "test.png"
        img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        mock_face_detector = MagicMock()
        mock_face_detector.detect.return_value = (5, 5, 30, 30)

        mock_landmarks = MagicMock()
        mock_landmarks.face_rect = (5, 5, 30, 30)
        mock_landmarks.landmarks_by_region = {
            "left_eyebrow": [(10, 20), (15, 20)],
            "right_eyebrow": [(25, 20), (30, 20)],
        }

        mock_landmark_detector = MagicMock()
        mock_landmark_detector.detect.return_value = mock_landmarks

        mock_hairline_result = {
            "hairline_y": 10,
            "method": "canny",
            "canny_edge_map": np.zeros((15, 30), dtype=np.uint8),
            "center_column": np.zeros(15, dtype=np.uint8),
        }
        mock_hairline_detector = MagicMock()
        mock_hairline_detector.detect.return_value = mock_hairline_result

        with patch("scripts.batch_canny_viewer.cv2.namedWindow"), \
             patch("scripts.batch_canny_viewer.cv2.imshow"), \
             patch("scripts.batch_canny_viewer.cv2.waitKey", return_value=0), \
             patch("scripts.batch_canny_viewer.cv2.destroyWindow"):
            result = bcv._process_image(
                img_path, "close", 3, 30, 100, output_dir, True,
                face_detector=mock_face_detector,
                landmark_detector=mock_landmark_detector,
                hairline_detector=mock_hairline_detector,
            )

        assert result is not None
        assert result.exists()
        mock_face_detector.detect.assert_called_once()
        mock_landmark_detector.detect.assert_called_once()
        mock_hairline_detector.detect.assert_called_once()

        # Verify parameters are passed through to hairline detector
        call_args = mock_hairline_detector.detect.call_args
        assert call_args[1]["canny_low"] == 30
        assert call_args[1]["canny_high"] == 100
        assert call_args[1]["close_ksize"] == 3


class TestMain:
    """Tests for the main entry point."""

    def test_no_images_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """An empty directory should print a message and exit cleanly."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch.object(
            sys, "argv", [
                "batch_canny_viewer.py",
                "--operation", "close",
                "--kernel-size", "3",
                "--input", str(empty_dir),
            ]
        ):
            with pytest.raises(SystemExit) as exc_info:
                bcv.main()
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "No images found" in captured.out

    def test_missing_directory(self, capsys: pytest.CaptureFixture) -> None:
        """A missing input directory should print an error and exit with code 1."""
        with patch.object(
            sys, "argv", [
                "batch_canny_viewer.py",
                "--operation", "close",
                "--kernel-size", "3",
                "--input", "nonexistent_dir_12345",
            ]
        ):
            with pytest.raises(SystemExit) as exc_info:
                bcv.main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    def test_summary_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """After processing, a summary should be printed."""
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        img_path = img_dir / "face.png"
        img = np.random.randint(
            0, 256, (40, 40, 3), dtype=np.uint8
        )
        cv2.imwrite(str(img_path), img)

        with patch.object(
            sys, "argv", [
                "batch_canny_viewer.py",
                "--operation", "close",
                "--kernel-size", "3",
                "--input", str(img_dir),
            ]
        ):
            bcv.main()

        captured = capsys.readouterr()
        assert "Batch Canny Edge Detection Summary" in captured.out
        assert "Images processed: 1" in captured.out
        assert "Operation: close" in captured.out
        assert "Kernel size: 3" in captured.out

    def test_detector_init_failure_graceful(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """If detector init fails, processing continues without hairline."""
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        img_path = img_dir / "face.png"
        img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        with patch.object(
            sys, "argv", [
                "batch_canny_viewer.py",
                "--operation", "close",
                "--kernel-size", "3",
                "--input", str(img_dir),
            ]
        ), patch(
            "scripts.batch_canny_viewer.ModelFinder.find",
            side_effect=Exception("Model not found"),
        ):
            bcv.main()

        captured = capsys.readouterr()
        assert "WARNING: Could not initialize detectors" in captured.out
        assert "Hairline overlay will be disabled" in captured.out
        assert "Batch Canny Edge Detection Summary" in captured.out
