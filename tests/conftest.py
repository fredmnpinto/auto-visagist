"""Shared fixtures for all visagism tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from visagism.types import FaceRect, FacialLandmarks, LandmarkRegions


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    """Create a valid 400x400 test image and return its path."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Fill with a light grey so it's not completely black
    img[:] = (200, 200, 200)
    path = tmp_path / "test_face.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def sample_image_rgb(sample_image_path: Path) -> np.ndarray:
    """Load and return the sample image as a BGR array."""
    return cv2.imread(str(sample_image_path), cv2.IMREAD_COLOR)


@pytest.fixture
def sample_image_gray(sample_image_path: Path) -> np.ndarray:
    """Load and return the sample image as a grayscale array."""
    img = cv2.imread(str(sample_image_path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def sample_face_rect() -> FaceRect:
    """Return a sample face bounding box."""
    return (50, 50, 200, 250)


@pytest.fixture
def sample_landmarks() -> FacialLandmarks:
    """Create sample FacialLandmarks with synthetic 68-point data."""
    # Generate 68 points in a rough face-like arrangement
    pts = [(100 + i, 100 + i // 2) for i in range(68)]
    landmarks_68 = list(pts)

    # Build regions
    from visagism.constants import REGION_INDICES

    landmarks_by_region: LandmarkRegions = {
        name: [landmarks_68[i] for i in indices]
        for name, indices in REGION_INDICES.items()
    }

    return FacialLandmarks(
        image_path=Path("/fake/test.jpg"),
        face_rect=(50, 50, 200, 250),
        landmarks_68=landmarks_68,
        landmarks_by_region=landmarks_by_region,
    )


@pytest.fixture
def small_image_path(tmp_path: Path) -> Path:
    """Create a too-small 100x100 image and return its path."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    path = tmp_path / "too_small.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def fake_model_path(tmp_path: Path) -> Path:
    """Create a fake (empty) model file."""
    path = tmp_path / "shape_predictor_68_face_landmarks.dat"
    path.touch()
    return path
