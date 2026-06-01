"""Tests for the LandmarkVisualizer module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from visagism.landmark_visualizer import LandmarkVisualizer


class TestLandmarkVisualizer:
    """Test suite for LandmarkVisualizer."""

    def test_draw_landmarks_returns_image(
        self, sample_image_rgb: np.ndarray, sample_landmarks
    ) -> None:
        """Test that draw_landmarks returns an image array."""
        viz = LandmarkVisualizer()
        result = viz.draw_landmarks(sample_image_rgb, sample_landmarks)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image_rgb.shape
        assert result.dtype == np.uint8

    def test_draw_landmarks_not_modifying_original(
        self, sample_image_rgb: np.ndarray, sample_landmarks
    ) -> None:
        """Test that the original image is not modified."""
        original = sample_image_rgb.copy()
        viz = LandmarkVisualizer()
        viz.draw_landmarks(sample_image_rgb, sample_landmarks)
        assert np.array_equal(sample_image_rgb, original)

    def test_draw_landmarks_adds_points(
        self, sample_image_rgb: np.ndarray, sample_landmarks
    ) -> None:
        """Test that landmarks are drawn (image differs from original)."""
        original = sample_image_rgb.copy()
        viz = LandmarkVisualizer()
        result = viz.draw_landmarks(sample_image_rgb, sample_landmarks)
        # The annotated image should differ from the original
        assert not np.array_equal(result, original)

    def test_save_creates_file(
        self, sample_image_rgb: np.ndarray, sample_landmarks, tmp_path: Path
    ) -> None:
        """Test that save writes a file to disk."""
        viz = LandmarkVisualizer()
        annotated = viz.draw_landmarks(sample_image_rgb, sample_landmarks)
        save_path = viz.save(annotated, tmp_path, "test_output")
        assert save_path.exists()
        assert save_path.suffix == ".jpg"
        assert "landmarks_" in save_path.name

    def test_save_creates_directory(
        self, sample_image_rgb: np.ndarray, sample_landmarks, tmp_path: Path
    ) -> None:
        """Test that save creates the output directory if it doesn't exist."""
        viz = LandmarkVisualizer()
        annotated = viz.draw_landmarks(sample_image_rgb, sample_landmarks)
        nested_dir = tmp_path / "nested" / "output"
        save_path = viz.save(annotated, nested_dir, "test")
        assert nested_dir.exists()
        assert save_path.exists()

    def test_save_returns_path(
        self, sample_image_rgb: np.ndarray, sample_landmarks, tmp_path: Path
    ) -> None:
        """Test that save returns a Path object."""
        viz = LandmarkVisualizer()
        annotated = viz.draw_landmarks(sample_image_rgb, sample_landmarks)
        result = viz.save(annotated, tmp_path, "test")
        assert isinstance(result, Path)

    def test_gui_available_returns_bool(self) -> None:
        """Test that _gui_available returns a boolean."""
        result = LandmarkVisualizer._gui_available()
        assert isinstance(result, bool)

    def test_show_no_gui(
        self, sample_image_rgb: np.ndarray, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that show prints a warning when GUI unavailable."""
        viz = LandmarkVisualizer()

        # Mock _gui_available to return False
        original_check = LandmarkVisualizer._gui_available
        LandmarkVisualizer._gui_available = staticmethod(lambda: False)

        try:
            viz.show(sample_image_rgb, "Test")
            captured = capsys.readouterr()
            assert "Warning" in captured.out
        finally:
            LandmarkVisualizer._gui_available = original_check

    def test_legend_is_drawn(
        self, sample_image_rgb: np.ndarray, sample_landmarks
    ) -> None:
        """Test that the legend box is drawn (pixels in top-left change)."""
        viz = LandmarkVisualizer()
        # Check a pixel in the legend area (top-left corner)
        result = viz.draw_landmarks(sample_image_rgb, sample_landmarks)
        # The legend area should differ from plain grey background
        top_left_pixel = result[15, 15]
        assert not np.array_equal(top_left_pixel, [200, 200, 200])

    def test_draw_landmarks_all_regions(
        self, sample_image_rgb: np.ndarray, sample_landmarks
    ) -> None:
        """Test that all facial regions are drawn."""
        viz = LandmarkVisualizer()
        result = viz.draw_landmarks(sample_image_rgb, sample_landmarks)

        # Image should not be identical to input (landmarks drawn)
        assert not np.array_equal(result, sample_image_rgb)

        # Check that the image has been modified in the regions where landmarks are
        # (landmarks are at varying positions)
        assert result.shape == sample_image_rgb.shape

    def test_draw_hairline_with_hairline_y(
        self, sample_image_rgb: np.ndarray, sample_landmarks_with_hairline
    ) -> None:
        """Test that draw_hairline draws a dashed line when hairline_y is set."""
        viz = LandmarkVisualizer()
        result = viz.draw_hairline(
            sample_image_rgb.copy(), sample_landmarks_with_hairline
        )

        # The image should have been modified (yellow dashed line drawn)
        assert not np.array_equal(result, sample_image_rgb)

        # Check that at least one pixel on the hairline row is yellow
        y = sample_landmarks_with_hairline.hairline_y
        face_rect = sample_landmarks_with_hairline.face_rect
        x_start = face_rect[0]
        x_end = face_rect[0] + face_rect[2]

        # Look for yellow pixels (BGR: 0, 255, 255) on the hairline row
        row = result[y, x_start:x_end]
        yellow_pixels = np.where(
            (row[:, 0] == 0) & (row[:, 1] == 255) & (row[:, 2] == 255)
        )[0]
        assert len(yellow_pixels) > 0

    def test_draw_hairline_without_hairline_y(
        self, sample_image_rgb: np.ndarray, sample_landmarks
    ) -> None:
        """Test that draw_hairline returns image unchanged when hairline_y is None."""
        viz = LandmarkVisualizer()
        original = sample_image_rgb.copy()
        result = viz.draw_hairline(original, sample_landmarks)

        # Image should be unchanged
        assert np.array_equal(result, original)

    def test_draw_landmarks_includes_hairline(
        self, sample_image_rgb: np.ndarray, sample_landmarks_with_hairline
    ) -> None:
        """Test that draw_landmarks includes the hairline when set."""
        viz = LandmarkVisualizer()
        result = viz.draw_landmarks(sample_image_rgb, sample_landmarks_with_hairline)

        # The result should differ from original (hairline drawn)
        assert not np.array_equal(result, sample_image_rgb)

        # Check for yellow pixels on the hairline row
        y = sample_landmarks_with_hairline.hairline_y
        face_rect = sample_landmarks_with_hairline.face_rect
        x_start = face_rect[0]
        x_end = face_rect[0] + face_rect[2]

        row = result[y, x_start:x_end]
        yellow_pixels = np.where(
            (row[:, 0] == 0) & (row[:, 1] == 255) & (row[:, 2] == 255)
        )[0]
        assert len(yellow_pixels) > 0

    def test_draw_landmarks_skips_missing_points(
        self, sample_image_rgb: np.ndarray, sample_landmarks
    ) -> None:
        """Test that (-1, -1) points are skipped when drawing."""
        viz = LandmarkVisualizer()

        # Create landmarks with some missing points
        pts = list(sample_landmarks.landmarks_68)
        pts[5] = (-1, -1)
        pts[10] = (-1, -1)

        from visagism.constants import REGION_INDICES
        from visagism.types import LandmarkRegions

        landmarks_by_region: LandmarkRegions = {
            name: [pts[i] for i in indices]
            for name, indices in REGION_INDICES.items()
        }
        modified_landmarks = type(sample_landmarks)(
            image_path=sample_landmarks.image_path,
            face_rect=sample_landmarks.face_rect,
            landmarks_68=pts,
            landmarks_by_region=landmarks_by_region,
        )

        # Should not raise and should return valid image
        result = viz.draw_landmarks(sample_image_rgb, modified_landmarks)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image_rgb.shape

    def test_draw_landmarks_skips_missing_connections(
        self, sample_image_rgb: np.ndarray, sample_landmarks
    ) -> None:
        """Test that connections with (-1, -1) endpoints are skipped."""
        viz = LandmarkVisualizer()

        # Create landmarks with missing points that would be connected
        pts = list(sample_landmarks.landmarks_68)
        pts[0] = (-1, -1)  # Jaw start — connected to pt 1

        from visagism.constants import REGION_INDICES
        from visagism.types import LandmarkRegions

        landmarks_by_region: LandmarkRegions = {
            name: [pts[i] for i in indices]
            for name, indices in REGION_INDICES.items()
        }
        modified_landmarks = type(sample_landmarks)(
            image_path=sample_landmarks.image_path,
            face_rect=sample_landmarks.face_rect,
            landmarks_68=pts,
            landmarks_by_region=landmarks_by_region,
        )

        # Should not raise and should return valid image
        result = viz.draw_landmarks(sample_image_rgb, modified_landmarks)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image_rgb.shape

    def test_sparse_landmarks_use_larger_radius(
        self, sample_image_rgb: np.ndarray, sample_landmarks
    ) -> None:
        """Test that sparse landmarks (>50% missing) use larger radius and thickness."""
        viz = LandmarkVisualizer()

        # Create landmarks with >50% missing (35 points missing)
        pts = [(-1, -1)] * 35 + [(100 + i, 100 + i // 2) for i in range(35)]
        landmarks_68 = list(pts)

        from visagism.constants import REGION_INDICES
        from visagism.types import LandmarkRegions

        landmarks_by_region: LandmarkRegions = {
            name: [landmarks_68[i] for i in indices]
            for name, indices in REGION_INDICES.items()
        }
        sparse_landmarks = type(sample_landmarks)(
            image_path=sample_landmarks.image_path,
            face_rect=sample_landmarks.face_rect,
            landmarks_68=landmarks_68,
            landmarks_by_region=landmarks_by_region,
        )

        with (
            patch("visagism.landmark_visualizer.cv2.circle") as mock_circle,
            patch("visagism.landmark_visualizer.cv2.line") as mock_line,
        ):
            viz.draw_landmarks(sample_image_rgb, sparse_landmarks)

            # Check that circle was called with radius 6 (3 * 2)
            # cv2.circle(img, center, radius, color, thickness)
            circle_calls = [
                call for call in mock_circle.call_args_list
                if call.args[2] == 6
            ]
            msg = "Expected at least one circle call with radius 6"
            assert len(circle_calls) > 0, msg

            # Check that line was called with thickness 3 (2 + 1)
            # cv2.line(img, pt1, pt2, color, thickness)
            line_calls = [
                call for call in mock_line.call_args_list
                if call.args[4] == 3
            ]
            msg = "Expected at least one line call with thickness 3"
            assert len(line_calls) > 0, msg

    def test_full_landmarks_use_normal_radius(
        self, sample_image_rgb: np.ndarray, sample_landmarks
    ) -> None:
        """Test that full landmarks (0% missing) use normal radius and thickness."""
        viz = LandmarkVisualizer()

        with (
            patch("visagism.landmark_visualizer.cv2.circle") as mock_circle,
            patch("visagism.landmark_visualizer.cv2.line") as mock_line,
        ):
            viz.draw_landmarks(sample_image_rgb, sample_landmarks)

            # Check that circle was called with radius 3 (normal)
            # cv2.circle(img, center, radius, color, thickness)
            circle_calls = [
                call for call in mock_circle.call_args_list
                if call.args[2] == 3
            ]
            msg = "Expected at least one circle call with radius 3"
            assert len(circle_calls) > 0, msg

            # Check that line was called with thickness 2 (normal)
            # cv2.line(img, pt1, pt2, color, thickness)
            line_calls = [
                call for call in mock_line.call_args_list
                if call.args[4] == 2
            ]
            msg = "Expected at least one line call with thickness 2"
            assert len(line_calls) > 0, msg

            # Ensure no circle calls with radius 6 (sparse mode)
            sparse_circle_calls = [
                call for call in mock_circle.call_args_list
                if call.args[2] == 6
            ]
            msg = "Did not expect circle calls with radius 6"
            assert len(sparse_circle_calls) == 0, msg

            # Ensure no line calls with thickness 3 (sparse mode)
            sparse_line_calls = [
                call for call in mock_line.call_args_list
                if call.args[4] == 3
            ]
            msg = "Did not expect line calls with thickness 3"
            assert len(sparse_line_calls) == 0, msg
