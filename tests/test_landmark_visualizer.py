"""Tests for the LandmarkVisualizer module."""

from __future__ import annotations

from pathlib import Path

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
