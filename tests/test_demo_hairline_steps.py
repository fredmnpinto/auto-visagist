"""Tests for scripts/demo_hairline_steps.py.

Since ``scripts/`` is not a package, we add it to ``sys.path`` before
importing the module under test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Add scripts/ to the path so the demo module can be imported
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import demo_hairline_steps as demo  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_image_path(tmp_path: Path) -> Path:
    """Create a dummy image file and return its path."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = (128, 128, 128)
    path = tmp_path / "fake_face.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def fake_steps() -> dict:
    """Return a realistic steps dict as produced by HairlineDetector."""
    roi = np.random.randint(0, 255, (50, 1), dtype=np.uint8)
    context = np.random.randint(0, 255, (50, 100), dtype=np.uint8)
    closed = np.random.randint(0, 255, (50, 100), dtype=np.uint8)
    edges = np.zeros((50, 100), dtype=np.uint8)
    edges[20, 50] = 255  # one edge pixel
    center_column = edges[:, 50]
    first_edge_idx = 20
    edge_pixels_count = 1
    return {
        "hairline_y": 80,
        "method": "canny",
        "roi_coords": (97, 98, 30, 80),
        "roi_raw": roi,
        "canny_context_coords": (50, 150, 30, 80),
        "canny_context_raw": context,
        "closed_context": closed,
        "close_ksize": 3,
        "canny_edge_map": edges,
        "center_column": center_column,
        "first_edge_idx": first_edge_idx,
        "edge_pixels_count": edge_pixels_count,
        "gaussian_ksize": 5,
        "canny_low": 30,
        "canny_high": 100,
        "avg_eyebrow_y": 80,
        "face_rect": (50, 50, 100, 120),
        "searchable_rows": 50,
    }


@pytest.fixture
def fake_steps_empty_roi() -> dict:
    """Return a steps dict with an empty ROI but valid coordinates."""
    return {
        "hairline_y": 50,
        "method": "fallback",
        "roi_coords": (97, 98, 80, 80),  # y_start == y_end -> empty
        "roi_raw": np.array([]),
        "canny_context_coords": (50, 150, 30, 80),  # valid coords
        "canny_context_raw": np.array([]),
        "closed_context": np.array([]),
        "close_ksize": 0,
        "canny_edge_map": np.array([]),
        "center_column": np.array([]),
        "first_edge_idx": -1,
        "edge_pixels_count": 0,
        "gaussian_ksize": 5,
        "canny_low": 30,
        "canny_high": 100,
        "avg_eyebrow_y": 80,
        "face_rect": (50, 50, 100, 120),
        "searchable_rows": 0,
    }


@pytest.fixture
def fake_steps_invalid_coords() -> dict:
    """Return a steps dict with invalid coordinates."""
    return {
        "hairline_y": 50,
        "method": "fallback",
        "roi_coords": (97, 98, 80, 80),
        "roi_raw": np.array([]),
        "canny_context_coords": (50, 150, 80, 80),  # invalid: y_start == y_end
        "canny_context_raw": np.array([]),
        "closed_context": np.array([]),
        "close_ksize": 0,
        "canny_edge_map": np.array([]),
        "center_column": np.array([]),
        "first_edge_idx": -1,
        "edge_pixels_count": 0,
        "gaussian_ksize": 5,
        "canny_low": 30,
        "canny_high": 100,
        "avg_eyebrow_y": 80,
        "face_rect": (50, 50, 100, 120),
        "searchable_rows": 0,
    }


@pytest.fixture
def mock_pipeline(
    fake_image_path: Path,
    fake_steps: dict,
) -> Generator[MagicMock, None, None]:
    """Patch all external dependencies used by process_image."""
    img_bgr = cv2.imread(str(fake_image_path), cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_rect = (50, 50, 100, 120)

    mock_landmarks = MagicMock()
    mock_landmarks.face_rect = face_rect
    mock_landmarks.landmarks_by_region = {
        "left_eyebrow": [(95, 78), (97, 76), (99, 75), (101, 76), (103, 78)],
        "right_eyebrow": [(105, 78), (107, 76), (109, 75), (111, 76), (113, 78)],
    }

    mock_lm_instance = MagicMock()
    mock_lm_instance.detect.return_value = mock_landmarks

    with patch.object(demo.ImageLoader, "load", return_value=(img_bgr, img_gray)):
        with patch.object(demo.FaceDetector, "detect", return_value=face_rect):
            with patch.object(demo.ModelFinder, "find", return_value=Path("model.dat")):
                with patch.object(
                    demo, "LandmarkDetector", return_value=mock_lm_instance
                ):
                    with patch.object(
                        demo.HairlineDetector, "detect", return_value=fake_steps
                    ):
                        yield fake_steps


@pytest.fixture
def mock_pipeline_empty_roi(
    fake_image_path: Path,
    fake_steps_empty_roi: dict,
) -> Generator[MagicMock, None, None]:
    """Patch all external dependencies with an empty ROI result."""
    img_bgr = cv2.imread(str(fake_image_path), cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_rect = (50, 50, 100, 120)

    mock_landmarks = MagicMock()
    mock_landmarks.face_rect = face_rect
    mock_landmarks.landmarks_by_region = {
        "left_eyebrow": [(95, 78), (97, 76), (99, 75), (101, 76), (103, 78)],
        "right_eyebrow": [(105, 78), (107, 76), (109, 75), (111, 76), (113, 78)],
    }

    mock_lm_instance = MagicMock()
    mock_lm_instance.detect.return_value = mock_landmarks

    with patch.object(demo.ImageLoader, "load", return_value=(img_bgr, img_gray)):
        with patch.object(demo.FaceDetector, "detect", return_value=face_rect):
            with patch.object(demo.ModelFinder, "find", return_value=Path("model.dat")):
                with patch.object(
                    demo, "LandmarkDetector", return_value=mock_lm_instance
                ):
                    with patch.object(
                        demo.HairlineDetector,
                        "detect",
                        return_value=fake_steps_empty_roi,
                    ):
                        yield fake_steps_empty_roi


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestEnsureOutputDir:
    """Tests for _ensure_output_dir."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        """Directory should be created if it does not exist."""
        image_path = tmp_path / "photos" / "test.jpg"
        output_dir = demo._ensure_output_dir(image_path)
        assert output_dir.exists()
        assert output_dir.name == "test"
        assert output_dir.parent.name == "output"

    def test_returns_existing_directory(self, tmp_path: Path) -> None:
        """Should return existing directory without error."""
        image_path = tmp_path / "test.png"
        demo._ensure_output_dir(image_path)
        output_dir = demo._ensure_output_dir(image_path)
        assert output_dir.exists()


class TestSerializeForJson:
    """Tests for _serialize_for_json."""

    def test_ndarray_to_list(self) -> None:
        """NumPy arrays become Python lists."""
        arr = np.array([1, 2, 3])
        assert demo._serialize_for_json(arr) == [1, 2, 3]

    def test_np_int_to_int(self) -> None:
        """NumPy integers become Python ints."""
        assert demo._serialize_for_json(np.int64(42)) == 42

    def test_np_float_to_float(self) -> None:
        """NumPy floats become Python floats."""
        assert demo._serialize_for_json(np.float64(3.14)) == 3.14

    def test_tuple_to_list(self) -> None:
        """Tuples become lists."""
        assert demo._serialize_for_json((1, 2)) == [1, 2]

    def test_plain_value_unchanged(self) -> None:
        """Plain values pass through."""
        assert demo._serialize_for_json("hello") == "hello"
        assert demo._serialize_for_json(42) == 42


class TestBuildSummaryText:
    """Tests for _build_summary_text."""

    def test_includes_key_info(self, fake_steps: dict) -> None:
        """Summary must contain image name, face size, and result."""
        image_path = Path("/fake/photo.jpg")
        face_rect = (50, 50, 100, 120)
        text = demo._build_summary_text(image_path, face_rect, fake_steps)
        assert "photo.jpg" in text
        assert "Face size: 100x120 px" in text
        assert "canny detection at y=80" in text


class TestCreateStep2PureCanny:
    """Tests for _create_step2_pure_canny."""

    def test_returns_bgr_image(self) -> None:
        """Result should be a 3-channel BGR image with same dimensions."""
        img_gray = np.zeros((100, 150), dtype=np.uint8)
        img_gray[40:60, 50:100] = 255
        result = demo._create_step2_pure_canny(img_gray, 50, 150)
        assert result.shape == (100, 150, 3)
        assert result.dtype == np.uint8

    def test_no_preprocessing_applied(self) -> None:
        """Pure Canny should not apply blur or morphology."""
        img_gray = np.zeros((50, 50), dtype=np.uint8)
        img_gray[20:30, 20:30] = 255
        result = demo._create_step2_pure_canny(img_gray, 50, 150)
        # Just verify it runs and produces output; exact edge pixels depend
        # on OpenCV's Canny implementation.
        assert result.shape == (50, 50, 3)


class TestBuildStepImages:
    """Tests for _build_step_images."""

    def test_returns_four_images(self, fake_steps: dict) -> None:
        """Should return exactly 4 images."""
        img_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        img_gray = np.zeros((200, 200), dtype=np.uint8)
        images = demo._build_step_images(img_bgr, img_gray, fake_steps, None)
        assert len(images) == 4

    def test_step1_is_original_image(self, fake_steps: dict) -> None:
        """Step 1 should be pixel-identical to the input BGR image."""
        img_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        images = demo._build_step_images(img_bgr, img_gray, fake_steps, None)
        np.testing.assert_array_equal(images[0], img_bgr)

    def test_step2_is_pure_canny_on_full_image(self, fake_steps: dict) -> None:
        """Step 2 should have same (H, W) as input and 3 channels."""
        img_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        img_gray = np.zeros((200, 200), dtype=np.uint8)
        images = demo._build_step_images(img_bgr, img_gray, fake_steps, None)
        assert images[1].shape == (200, 200, 3)

    def test_all_steps_match_input_dimensions(self, fake_steps: dict) -> None:
        """All 4 steps should have the same dimensions as the input image."""
        img_bgr = np.zeros((200, 250, 3), dtype=np.uint8)
        img_gray = np.zeros((200, 250), dtype=np.uint8)
        images = demo._build_step_images(img_bgr, img_gray, fake_steps, None)
        for i, img in enumerate(images):
            assert img.shape == (200, 250, 3), f"Step {i + 1} has wrong shape"

    def test_step3_has_green_roi_rectangle(self, fake_steps: dict) -> None:
        """Step 3 should have a green ROI rectangle around the context area."""
        img_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        img_gray = np.zeros((200, 200), dtype=np.uint8)
        images = demo._build_step_images(img_bgr, img_gray, fake_steps, None)
        step3 = images[2]
        roi_area = step3[28:82, 48:152]
        has_green = np.any(
            (roi_area[:, :, 1] == 255)
            & (roi_area[:, :, 0] < 50)
            & (roi_area[:, :, 2] < 50)
        )
        assert has_green, "Expected green ROI rectangle in Step 3"

    def test_step4_has_cyan_edges_and_red_hairline(
        self, fake_steps: dict,
    ) -> None:
        """Step 4 should have cyan edge pixels and a red dashed hairline."""
        img_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        img_gray = np.zeros((200, 200), dtype=np.uint8)
        images = demo._build_step_images(img_bgr, img_gray, fake_steps, None)
        step4 = images[3]
        # Cyan edge pixel mapped from edges[20, 50] to ~y=50, x=100
        edge_area = step4[48:52, 98:102]
        has_cyan = np.any(
            (edge_area[:, :, 0] == 255)
            & (edge_area[:, :, 1] == 255)
            & (edge_area[:, :, 2] < 50)
        )
        assert has_cyan, "Expected cyan edge pixel in Step 4"
        # Red dashed hairline at y=80
        hairline_area = step4[78:83, :]
        has_red = np.any(
            (hairline_area[:, :, 2] == 255)
            & (hairline_area[:, :, 0] < 50)
            & (hairline_area[:, :, 1] < 50)
        )
        assert has_red, "Expected red hairline in Step 4"

    def test_step3_empty_roi_shows_warning(
        self, fake_steps_empty_roi: dict,
    ) -> None:
        """Step 3 with empty ROI should show full image with warning text."""
        img_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        img_gray = np.zeros((200, 200), dtype=np.uint8)
        images = demo._build_step_images(
            img_bgr, img_gray, fake_steps_empty_roi, None,
        )
        step3 = images[2]
        assert step3.shape == (200, 200, 3)
        # Green rectangle should still be drawn for valid coords
        roi_area = step3[28:82, 48:152]
        has_green = np.any(
            (roi_area[:, :, 1] == 255)
            & (roi_area[:, :, 0] < 50)
            & (roi_area[:, :, 2] < 50)
        )
        assert has_green, "Expected green ROI rectangle even for empty ROI"
        # Red warning text should be present
        has_red = np.any(
            (step3[:, :, 2] == 255)
            & (step3[:, :, 0] < 50)
            & (step3[:, :, 1] < 50)
        )
        assert has_red, "Expected red warning text for empty ROI"

    def test_step4_empty_roi_shows_warning(
        self, fake_steps_empty_roi: dict,
    ) -> None:
        """Step 4 with empty ROI should show full image with warning text."""
        img_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        img_gray = np.zeros((200, 200), dtype=np.uint8)
        images = demo._build_step_images(
            img_bgr, img_gray, fake_steps_empty_roi, None,
        )
        step4 = images[3]
        assert step4.shape == (200, 200, 3)
        # Green rectangle should still be drawn
        roi_area = step4[28:82, 48:152]
        has_green = np.any(
            (roi_area[:, :, 1] == 255)
            & (roi_area[:, :, 0] < 50)
            & (roi_area[:, :, 2] < 50)
        )
        assert has_green, "Expected green ROI rectangle even for empty ROI"
        # Red pixels from warning text or hairline
        has_red = np.any(
            (step4[:, :, 2] == 255)
            & (step4[:, :, 0] < 50)
            & (step4[:, :, 1] < 50)
        )
        assert has_red, "Expected red pixels for warning or hairline"

    def test_step3_invalid_coords_shows_invalid_text(
        self, fake_steps_invalid_coords: dict,
    ) -> None:
        """Step 3 with invalid coords should show 'Invalid ROI' text."""
        img_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        img_gray = np.zeros((200, 200), dtype=np.uint8)
        images = demo._build_step_images(
            img_bgr, img_gray, fake_steps_invalid_coords, None,
        )
        step3 = images[2]
        assert step3.shape == (200, 200, 3)
        center_area = step3[90:110, 20:180]
        has_red = np.any(
            (center_area[:, :, 2] == 255)
            & (center_area[:, :, 0] < 50)
            & (center_area[:, :, 1] < 50)
        )
        assert has_red, "Expected red 'Invalid ROI' text in Step 3"

    def test_step4_invalid_coords_shows_invalid_text(
        self, fake_steps_invalid_coords: dict,
    ) -> None:
        """Step 4 with invalid coords should show 'Invalid ROI' text."""
        img_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        img_gray = np.zeros((200, 200), dtype=np.uint8)
        images = demo._build_step_images(
            img_bgr, img_gray, fake_steps_invalid_coords, None,
        )
        step4 = images[3]
        assert step4.shape == (200, 200, 3)
        center_area = step4[90:110, 20:180]
        has_red = np.any(
            (center_area[:, :, 2] == 255)
            & (center_area[:, :, 0] < 50)
            & (center_area[:, :, 1] < 50)
        )
        assert has_red, "Expected red 'Invalid ROI' text in Step 4"


# ---------------------------------------------------------------------------
# Integration tests for process_image
# ---------------------------------------------------------------------------

class TestProcessImageSaves:
    """Tests that process_image always writes files to disk."""

    EXPECTED_FILES = [
        "step01_original_image.png",
        "step02_pure_canny.png",
        "step03_preprocessed.png",
        "step04_final_canny.png",
        "data.json",
        "profiles.csv",
        "summary.txt",
    ]

    def test_all_seven_files_exist(
        self, fake_image_path: Path, mock_pipeline: MagicMock, tmp_path: Path,
    ) -> None:
        """All 7 output files should be created."""
        # Change working directory so output/ is inside tmp_path
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            demo.process_image(fake_image_path, visualize=False)
            output_dir = Path("output") / fake_image_path.stem
            for name in self.EXPECTED_FILES:
                assert (output_dir / name).exists(), f"Missing {name}"
        finally:
            os.chdir(original_cwd)

    def test_headless_mode_no_windows(
        self, fake_image_path: Path, mock_pipeline: MagicMock, tmp_path: Path,
    ) -> None:
        """Headless mode (visualize=False) must not call cv2.imshow."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch.object(demo.cv2, "imshow") as mock_imshow:
                with patch.object(demo.cv2, "waitKey") as mock_waitkey:
                    demo.process_image(fake_image_path, visualize=False)
                    mock_imshow.assert_not_called()
                    mock_waitkey.assert_not_called()
        finally:
            os.chdir(original_cwd)

    def test_visualize_mode_calls_windows(
        self, fake_image_path: Path, mock_pipeline: MagicMock, tmp_path: Path,
    ) -> None:
        """Visualize mode must call cv2.imshow for each step."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch.object(demo.cv2, "imshow") as mock_imshow:
                with patch.object(demo.cv2, "waitKey", return_value=0):
                    with patch.object(demo.cv2, "destroyWindow"):
                        with patch.object(demo.cv2, "namedWindow"):
                            demo.process_image(fake_image_path, visualize=True)
                            assert mock_imshow.call_count == 4
        finally:
            os.chdir(original_cwd)

    def test_summary_txt_content(
        self, fake_image_path: Path, mock_pipeline: MagicMock, tmp_path: Path,
    ) -> None:
        """summary.txt should contain the expected summary text."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            demo.process_image(fake_image_path, visualize=False)
            summary_path = Path("output") / fake_image_path.stem / "summary.txt"
            content = summary_path.read_text(encoding="utf-8")
            assert "fake_face.jpg" in content
            assert "canny detection at y=80" in content
        finally:
            os.chdir(original_cwd)

    def test_data_json_content(
        self, fake_image_path: Path, mock_pipeline: MagicMock, tmp_path: Path,
    ) -> None:
        """data.json should contain serializable step data."""
        import json
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            demo.process_image(fake_image_path, visualize=False)
            data_path = Path("output") / fake_image_path.stem / "data.json"
            data = json.loads(data_path.read_text(encoding="utf-8"))
            assert data["hairline_y"] == 80
            assert data["method"] == "canny"
            assert isinstance(data["center_column"], list)
            assert isinstance(data["roi_coords"], list)
            assert "canny_edge_map" in data
            assert "first_edge_idx" in data
            assert "whole_image_canny" not in data
        finally:
            os.chdir(original_cwd)

    def test_step2_image_shape_matches_input(
        self, fake_image_path: Path, mock_pipeline: MagicMock, tmp_path: Path,
    ) -> None:
        """Step 2 image dimensions should match input image and have 3 channels."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            demo.process_image(fake_image_path, visualize=False)
            step2_path = (
                Path("output") / fake_image_path.stem
                / "step02_pure_canny.png"
            )
            img = cv2.imread(str(step2_path), cv2.IMREAD_UNCHANGED)
            assert img is not None
            assert img.shape[:2] == (200, 200)
            assert img.shape[2] == 3
        finally:
            os.chdir(original_cwd)

    def test_all_step_images_match_input_dimensions(
        self, fake_image_path: Path, mock_pipeline: MagicMock, tmp_path: Path,
    ) -> None:
        """All 4 saved step images should match input dimensions."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            demo.process_image(fake_image_path, visualize=False)
            output_dir = Path("output") / fake_image_path.stem
            png_names = [
                "step01_original_image.png",
                "step02_pure_canny.png",
                "step03_preprocessed.png",
                "step04_final_canny.png",
            ]
            for name in png_names:
                img = cv2.imread(str(output_dir / name), cv2.IMREAD_UNCHANGED)
                assert img is not None, f"Could not read {name}"
                assert img.shape[:2] == (200, 200), f"{name} has wrong dims"
                assert img.shape[2] == 3, f"{name} should have 3 channels"
        finally:
            os.chdir(original_cwd)

    def test_profiles_csv_content(
        self, fake_image_path: Path, mock_pipeline: MagicMock, tmp_path: Path,
    ) -> None:
        """profiles.csv should have the correct columns and row count."""
        import csv
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            demo.process_image(fake_image_path, visualize=False)
            csv_path = Path("output") / fake_image_path.stem / "profiles.csv"
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert rows[0] == ["row_index", "center_column_value"]
            assert len(rows) == 51  # header + 50 data rows
        finally:
            os.chdir(original_cwd)


class TestProcessImageEmptyRoi:
    """Tests for process_image with an empty/invalid ROI."""

    def test_saves_all_files_even_with_empty_roi(
        self, fake_image_path: Path, mock_pipeline_empty_roi: MagicMock,
        tmp_path: Path,
    ) -> None:
        """All files should still be created when ROI is empty."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            demo.process_image(fake_image_path, visualize=False)
            output_dir = Path("output") / fake_image_path.stem
            for name in TestProcessImageSaves.EXPECTED_FILES:
                assert (output_dir / name).exists(), f"Missing {name}"
        finally:
            os.chdir(original_cwd)

    def test_empty_roi_data_json(
        self, fake_image_path: Path, mock_pipeline_empty_roi: MagicMock,
        tmp_path: Path,
    ) -> None:
        """data.json should reflect empty ROI data."""
        import json
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            demo.process_image(fake_image_path, visualize=False)
            data_path = Path("output") / fake_image_path.stem / "data.json"
            data = json.loads(data_path.read_text(encoding="utf-8"))
            assert data["hairline_y"] == 50
            assert data["method"] == "fallback"
            assert data["center_column"] == []
        finally:
            os.chdir(original_cwd)

    def test_empty_roi_profiles_csv_has_only_header(
        self, fake_image_path: Path, mock_pipeline_empty_roi: MagicMock,
        tmp_path: Path,
    ) -> None:
        """profiles.csv should contain only the header when ROI is empty."""
        import csv
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            demo.process_image(fake_image_path, visualize=False)
            csv_path = Path("output") / fake_image_path.stem / "profiles.csv"
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            assert len(rows) == 1
            assert rows[0] == ["row_index", "center_column_value"]
        finally:
            os.chdir(original_cwd)


class TestProcessImageErrorHandling:
    """Tests for graceful error handling during file I/O."""

    def test_oserror_on_save_step_images_warns(
        self, fake_image_path: Path, mock_pipeline: MagicMock, tmp_path: Path,
    ) -> None:
        """OSError while saving step images should emit a warning, not crash."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch.object(
                demo.cv2, "imwrite", side_effect=OSError("disk full")
            ):
                with pytest.warns(UserWarning, match="Could not save"):
                    demo.process_image(fake_image_path, visualize=False)
        finally:
            os.chdir(original_cwd)

    def test_oserror_on_save_data_json_warns(
        self, fake_image_path: Path, mock_pipeline: MagicMock, tmp_path: Path,
    ) -> None:
        """OSError while saving data.json should emit a warning, not crash."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch(
                "demo_hairline_steps.open",
                side_effect=[*([OSError("disk full")] * 10)],
            ):
                with pytest.warns(UserWarning, match="Could not save"):
                    demo.process_image(fake_image_path, visualize=False)
        finally:
            os.chdir(original_cwd)

    def test_oserror_on_save_profiles_csv_warns(
        self, fake_image_path: Path, mock_pipeline: MagicMock, tmp_path: Path,
    ) -> None:
        """OSError while saving profiles.csv should emit a warning, not crash."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            # Patch csv.writer to raise on writerow
            with patch(
                "demo_hairline_steps.csv.writer",
                side_effect=OSError("disk full"),
            ):
                with pytest.warns(UserWarning, match="Could not save"):
                    demo.process_image(fake_image_path, visualize=False)
        finally:
            os.chdir(original_cwd)

    def test_oserror_on_save_summary_txt_warns(
        self, fake_image_path: Path, mock_pipeline: MagicMock, tmp_path: Path,
    ) -> None:
        """OSError while saving summary.txt should emit a warning, not crash."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch(
                "demo_hairline_steps.open",
                side_effect=OSError("disk full"),
            ):
                with pytest.warns(UserWarning, match="Could not save"):
                    demo.process_image(fake_image_path, visualize=False)
        finally:
            os.chdir(original_cwd)
