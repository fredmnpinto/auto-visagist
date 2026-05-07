"""Tests for the ImageLoader module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from visagism.errors import CorruptedImageError, ImageError, UnsupportedFormatError
from visagism.image_loader import ImageLoader
from visagism.constants import MIN_RESOLUTION


class TestImageLoader:
    """Test suite for ImageLoader.load()."""

    def test_load_valid_jpg(self, sample_image_path: Path) -> None:
        """Test loading a valid JPG image."""
        img_bgr, img_gray = ImageLoader.load(sample_image_path)
        assert img_bgr is not None
        assert img_gray is not None
        assert img_bgr.shape == (400, 400, 3)
        assert img_gray.shape == (400, 400)
        assert len(img_gray.shape) == 2  # grayscale

    def test_load_valid_png(self, tmp_path: Path) -> None:
        """Test loading a valid PNG image."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)
        img_bgr, img_gray = ImageLoader.load(path)
        assert img_bgr.shape == (300, 300, 3)

    def test_load_nonexistent_file(self) -> None:
        """Test loading a non-existent file raises ImageError."""
        with pytest.raises(ImageError, match="File not found"):
            ImageLoader.load(Path("/nonexistent/photo.jpg"))

    def test_load_unsupported_format(self, tmp_path: Path) -> None:
        """Test loading a .gif file raises UnsupportedFormatError."""
        path = tmp_path / "test.gif"
        path.touch()
        with pytest.raises(UnsupportedFormatError, match="Unsupported format"):
            ImageLoader.load(path)

    def test_load_unsupported_format_bmp(self, tmp_path: Path) -> None:
        """Test loading a .bmp file raises UnsupportedFormatError."""
        path = tmp_path / "test.bmp"
        path.touch()
        with pytest.raises(UnsupportedFormatError):
            ImageLoader.load(path)

    def test_load_corrupted_file(self, tmp_path: Path) -> None:
        """Test loading a corrupted file raises CorruptedImageError."""
        path = tmp_path / "corrupted.jpg"
        path.write_bytes(b"not a real image file")
        with pytest.raises(CorruptedImageError, match="Cannot read image file"):
            ImageLoader.load(path)

    def test_load_small_image(self, small_image_path: Path) -> None:
        """Test loading an image below minimum resolution prints warning."""
        img_bgr, img_gray = ImageLoader.load(small_image_path)
        assert img_bgr is not None
        assert img_gray is not None
        # Should still load successfully despite low resolution
        h, w = img_bgr.shape[:2]
        assert h < MIN_RESOLUTION[1] or w < MIN_RESOLUTION[0]

    def test_load_bgra_image(self, tmp_path: Path) -> None:
        """Test loading a BGRA image (with alpha channel) is converted to BGR."""
        img_bgra = np.zeros((300, 300, 4), dtype=np.uint8)
        path = tmp_path / "test_alpha.png"
        cv2.imwrite(str(path), img_bgra)
        img_bgr, img_gray = ImageLoader.load(path)
        assert img_bgr.shape[2] == 3  # BGR, not BGRA

    def test_load_case_insensitive_extension(self, tmp_path: Path) -> None:
        """Test that .JPG and .PNG extensions work (case insensitive)."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        path = tmp_path / "test.JPG"
        cv2.imwrite(str(path), img)
        img_bgr, img_gray = ImageLoader.load(path)
        assert img_bgr is not None

    def test_load_returns_correct_types(self, sample_image_path: Path) -> None:
        """Test that load returns numpy arrays."""
        img_bgr, img_gray = ImageLoader.load(sample_image_path)
        assert isinstance(img_bgr, np.ndarray)
        assert isinstance(img_gray, np.ndarray)
        assert img_bgr.dtype == np.uint8
        assert img_gray.dtype == np.uint8
