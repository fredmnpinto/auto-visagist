"""Unit tests for visagism.landmark_ground_truth module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visagism.errors import GroundTruthError
from visagism.landmark_ground_truth import LandmarkGroundTruth


class TestLandmarkGroundTruth:
    """Tests for the LandmarkGroundTruth dataclass."""

    def test_init_valid(self) -> None:
        """Test creation with valid 68 landmarks."""
        landmarks = [(i, i + 1) for i in range(68)]
        gt = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=1920,
            image_height=1080,
            landmarks_68=landmarks,
        )
        assert gt.image_width == 1920
        assert gt.image_height == 1080
        assert len(gt.landmarks_68) == 68

    def test_init_invalid_landmark_count(self) -> None:
        """Test that wrong landmark count raises GroundTruthError."""
        with pytest.raises(GroundTruthError, match="Expected exactly 68"):
            LandmarkGroundTruth(
                image_path=Path("/tmp/test.jpg"),
                image_width=100,
                image_height=100,
                landmarks_68=[(0, 0)] * 10,
            )

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        landmarks = [(i, i + 1) for i in range(68)]
        gt = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=1920,
            image_height=1080,
            landmarks_68=landmarks,
            hairline_y=312,
            corrected_landmarks=[0, 5, 10],
        )
        d = gt.to_dict()
        assert d["image_path"] == "/tmp/test.jpg"
        assert d["image_width"] == 1920
        assert d["image_height"] == 1080
        assert d["hairline_y"] == 312
        assert d["corrected_landmarks"] == [0, 5, 10]
        assert len(d["landmarks_68"]) == 68

    def test_to_json(self) -> None:
        """Test serialization to JSON string."""
        gt = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=[(-1, -1)] * 68,
        )
        json_str = gt.to_json()
        data = json.loads(json_str)
        assert data["image_path"] == "/tmp/test.jpg"
        assert data["image_width"] == 100

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving to file and loading back."""
        landmarks = [(i, i * 2) for i in range(68)]
        gt = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=800,
            image_height=600,
            landmarks_68=landmarks,
            hairline_y=150,
            corrected_landmarks=[2, 4, 6],
        )
        path = tmp_path / "test_gt.json"
        gt.save(path)
        assert path.exists()

        loaded = LandmarkGroundTruth.load(path)
        assert loaded.image_width == 800
        assert loaded.image_height == 600
        assert loaded.hairline_y == 150
        assert loaded.landmarks_68 == landmarks
        assert loaded.corrected_landmarks == [2, 4, 6]

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Test loading a non-existent file raises GroundTruthError."""
        with pytest.raises(GroundTruthError, match="Failed to load"):
            LandmarkGroundTruth.load(tmp_path / "nonexistent.json")

    def test_from_dict_missing_fields(self) -> None:
        """Test from_dict with missing required fields."""
        with pytest.raises(GroundTruthError, match="Missing required fields"):
            LandmarkGroundTruth.from_dict({"image_path": "/tmp/test.jpg"})

    def test_from_dict_pads_short_landmarks(self) -> None:
        """Test that short landmark lists are padded to 68."""
        data = {
            "image_path": "/tmp/test.jpg",
            "image_width": 100,
            "image_height": 100,
            "landmarks_68": [[0, 0], [1, 1]],
        }
        gt = LandmarkGroundTruth.from_dict(data)
        assert len(gt.landmarks_68) == 68
        assert gt.landmarks_68[2] == (-1, -1)
        assert gt.landmarks_68[0] == (0, 0)
        assert gt.corrected_landmarks == []

    def test_from_dict_trims_long_landmarks(self) -> None:
        """Test that long landmark lists are trimmed to 68."""
        data = {
            "image_path": "/tmp/test.jpg",
            "image_width": 100,
            "image_height": 100,
            "landmarks_68": [[i, i] for i in range(100)],
        }
        gt = LandmarkGroundTruth.from_dict(data)
        assert len(gt.landmarks_68) == 68

    def test_from_dict_with_corrected_landmarks(self) -> None:
        """Test that corrected_landmarks are parsed correctly."""
        data = {
            "image_path": "/tmp/test.jpg",
            "image_width": 100,
            "image_height": 100,
            "landmarks_68": [[0, 0]] * 68,
            "corrected_landmarks": [0, 5, 12],
        }
        gt = LandmarkGroundTruth.from_dict(data)
        assert gt.corrected_landmarks == [0, 5, 12]

    def test_from_dict_invalid_landmark_entries(self) -> None:
        """Test that invalid landmark entries become (-1, -1)."""
        data = {
            "image_path": "/tmp/test.jpg",
            "image_width": 100,
            "image_height": 100,
            "landmarks_68": [[0, 0], "invalid", [1, 1, 1], []],
        }
        gt = LandmarkGroundTruth.from_dict(data)
        assert gt.landmarks_68[0] == (0, 0)
        assert gt.landmarks_68[1] == (-1, -1)
        assert gt.landmarks_68[2] == (-1, -1)
        assert gt.landmarks_68[3] == (-1, -1)

    def test_from_json_invalid(self) -> None:
        """Test from_json with invalid JSON raises GroundTruthError."""
        with pytest.raises(GroundTruthError, match="Invalid JSON"):
            LandmarkGroundTruth.from_json("not json {")

    def test_get_region_for_landmark(self) -> None:
        """Test region lookup for landmark indices."""
        gt = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=[(-1, -1)] * 68,
        )
        assert gt.get_region_for_landmark(0) == "jaw"
        assert gt.get_region_for_landmark(16) == "jaw"
        assert gt.get_region_for_landmark(17) == "left_eyebrow"
        assert gt.get_region_for_landmark(36) == "left_eye"
        assert gt.get_region_for_landmark(67) == "inner_mouth"
        assert gt.get_region_for_landmark(100) == "unknown"

    def test_is_landmark_placed(self) -> None:
        """Test landmark placement check."""
        landmarks = [(-1, -1)] * 68
        landmarks[5] = (100, 200)
        gt = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=landmarks,
        )
        assert not gt.is_landmark_placed(0)
        assert gt.is_landmark_placed(5)
        assert not gt.is_landmark_placed(67)
        assert not gt.is_landmark_placed(100)  # out of range

    def test_completion_count(self) -> None:
        """Test completion counting."""
        landmarks = [(-1, -1)] * 68
        landmarks[0] = (1, 1)
        landmarks[10] = (2, 2)
        landmarks[20] = (3, 3)
        gt = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=landmarks,
        )
        placed, total = gt.completion_count()
        assert placed == 3
        assert total == 68

    def test_default_output_path(self) -> None:
        """Test default output path generation."""
        gt = LandmarkGroundTruth(
            image_path=Path("/photos/subject_01.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=[(-1, -1)] * 68,
        )
        path = gt.default_output_path()
        assert path.name == "subject_01_gt.json"
        assert path.parent.name == "ground_truth"
        assert path.parent.parent.name == "data"

    def test_default_output_path_custom_dir(self, tmp_path: Path) -> None:
        """Test default output path with custom directory."""
        gt = LandmarkGroundTruth(
            image_path=Path("/photos/subject_01.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=[(-1, -1)] * 68,
        )
        path = gt.default_output_path(output_dir=tmp_path)
        assert path == tmp_path / "subject_01_gt.json"

    def test_hairline_optional(self) -> None:
        """Test that hairline_y can be None."""
        gt = LandmarkGroundTruth(
            image_path=Path("/tmp/test.jpg"),
            image_width=100,
            image_height=100,
            landmarks_68=[(-1, -1)] * 68,
            hairline_y=None,
        )
        assert gt.hairline_y is None
        d = gt.to_dict()
        assert d["hairline_y"] is None
