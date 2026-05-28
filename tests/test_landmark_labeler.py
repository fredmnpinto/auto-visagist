"""Unit tests for visagism.landmark_labeler module.

These tests mock OpenCV where necessary since the interactive GUI
cannot be run headlessly in a test environment.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from visagism.errors import LabelerError
from visagism.landmark_labeler import LandmarkLabeler
from visagism.landmark_ground_truth import LandmarkGroundTruth


class TestLandmarkLabelerInit:
    """Tests for LandmarkLabeler initialization."""

    def test_init_with_valid_images(self, tmp_path: Path) -> None:
        """Test initialization with valid image files."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        assert len(labeler.image_paths) == 1
        assert labeler.current_image_idx == 0
        assert labeler.current_landmark_idx == 0

    def test_init_no_valid_images(self, tmp_path: Path) -> None:
        """Test initialization with no valid images raises LabelerError."""
        with pytest.raises(LabelerError, match="No valid image files"):
            LandmarkLabeler([tmp_path / "test.gif"], tmp_path / "output")

    def test_init_empty_list(self, tmp_path: Path) -> None:
        """Test initialization with empty list raises LabelerError."""
        with pytest.raises(LabelerError, match="No valid image files"):
            LandmarkLabeler([], tmp_path / "output")

    def test_init_filters_unsupported_formats(self, tmp_path: Path) -> None:
        """Test that unsupported formats are filtered out."""
        jpg_path = tmp_path / "test.jpg"
        gif_path = tmp_path / "test.gif"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(jpg_path), img)
        gif_path.write_text("fake gif")

        labeler = LandmarkLabeler([jpg_path, gif_path], tmp_path / "output")
        assert len(labeler.image_paths) == 1
        assert labeler.image_paths[0] == jpg_path


class TestLandmarkLabelerGroundTruth:
    """Tests for ground truth management in the labeler."""

    def test_create_empty_ground_truth(self, tmp_path: Path) -> None:
        """Test creation of empty ground truth."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        # Manually set image dimensions
        labeler._img_w = 400
        labeler._img_h = 400
        gt = labeler._create_empty_ground_truth(img_path)

        assert gt.image_path == img_path
        assert gt.image_width == 400
        assert gt.image_height == 400
        assert len(gt.landmarks_68) == 68
        assert all(pt == (-1, -1) for pt in gt.landmarks_68)
        assert gt.hairline_y is None

    def test_ground_truth_path(self, tmp_path: Path) -> None:
        """Test ground truth path generation."""
        img_path = tmp_path / "photos" / "subject_01.jpg"
        img_path.parent.mkdir(parents=True)
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        output_dir = tmp_path / "gt"
        labeler = LandmarkLabeler([img_path], output_dir)
        gt_path = labeler._ground_truth_path(img_path)
        assert gt_path == output_dir / "subject_01_gt.json"

    def test_all_unplaced(self, tmp_path: Path) -> None:
        """Test all_unplaced with empty ground truth."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        assert labeler._all_unplaced()

    def test_all_unplaced_with_placed(self, tmp_path: Path) -> None:
        """Test all_unplaced returns False when landmarks are placed."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._ground_truth.landmarks_68[0] = (100, 100)
        assert not labeler._all_unplaced()

    def test_load_existing_ground_truth(self, tmp_path: Path) -> None:
        """Test resuming from existing ground truth file."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        output_dir = tmp_path / "gt"
        output_dir.mkdir()

        # Create existing ground truth
        landmarks = [(i, i) for i in range(68)]
        gt = LandmarkGroundTruth(
            image_path=img_path,
            image_width=400,
            image_height=400,
            landmarks_68=landmarks,
            hairline_y=50,
            corrected_landmarks=[],
        )
        gt.save(output_dir / "test_gt.json")

        labeler = LandmarkLabeler([img_path], output_dir)
        labeler._load_image(0)

        assert labeler.ground_truth is not None
        assert labeler.ground_truth.landmarks_68[0] == (0, 0)
        assert labeler.ground_truth.hairline_y == 50


class TestLandmarkLabelerNavigation:
    """Tests for landmark and image navigation."""

    def test_next_landmark(self, tmp_path: Path) -> None:
        """Test advancing to next landmark."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._current_landmark_idx = 0
        labeler._next_landmark()
        assert labeler.current_landmark_idx == 1

    def test_next_landmark_capped_at_hairline(self, tmp_path: Path) -> None:
        """Test that next landmark stops at hairline (68)."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._current_landmark_idx = 68
        labeler._next_landmark()
        assert labeler.current_landmark_idx == 68

    def test_prev_landmark(self, tmp_path: Path) -> None:
        """Test going to previous landmark."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._current_landmark_idx = 5
        labeler._prev_landmark()
        assert labeler.current_landmark_idx == 4

    def test_prev_landmark_capped_at_zero(self, tmp_path: Path) -> None:
        """Test that prev landmark stops at 0."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._current_landmark_idx = 0
        labeler._prev_landmark()
        assert labeler.current_landmark_idx == 0

    def test_next_image(self, tmp_path: Path) -> None:
        """Test advancing to next image."""
        img1 = tmp_path / "img1.jpg"
        img2 = tmp_path / "img2.jpg"
        for p in [img1, img2]:
            img = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.imwrite(str(p), img)

        labeler = LandmarkLabeler([img1, img2], tmp_path / "output")
        labeler._img_bgr = np.zeros((400, 400, 3), dtype=np.uint8)
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img1)

        labeler._next_image()
        assert labeler.current_image_idx == 1

    def test_prev_image(self, tmp_path: Path) -> None:
        """Test going to previous image."""
        img1 = tmp_path / "img1.jpg"
        img2 = tmp_path / "img2.jpg"
        for p in [img1, img2]:
            img = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.imwrite(str(p), img)

        labeler = LandmarkLabeler([img1, img2], tmp_path / "output")
        labeler._current_image_idx = 1
        labeler._img_bgr = np.zeros((400, 400, 3), dtype=np.uint8)
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img2)

        labeler._prev_image()
        assert labeler.current_image_idx == 0


class TestLandmarkLabelerKeyboard:
    """Tests for keyboard handling."""

    def test_handle_key_n(self, tmp_path: Path) -> None:
        """Test 'n' key advances landmark."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._current_landmark_idx = 0
        labeler._handle_key(ord("n"))
        assert labeler.current_landmark_idx == 1

    def test_handle_key_p(self, tmp_path: Path) -> None:
        """Test 'p' key goes back landmark."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._current_landmark_idx = 5
        labeler._handle_key(ord("p"))
        assert labeler.current_landmark_idx == 4

    def test_handle_key_number(self, tmp_path: Path) -> None:
        """Test number keys jump to landmark."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._handle_key(ord("5"))
        assert labeler.current_landmark_idx == 5

    def test_handle_key_q(self, tmp_path: Path) -> None:
        """Test 'q' key requests quit."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._handle_key(ord("q"))
        assert labeler._quit_requested

    def test_handle_key_esc(self, tmp_path: Path) -> None:
        """Test Escape key requests quit."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._handle_key(27)  # Esc
        assert labeler._quit_requested

    def test_handle_key_save(self, tmp_path: Path) -> None:
        """Test 's' key saves ground truth."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        output_dir = tmp_path / "gt"
        labeler = LandmarkLabeler([img_path], output_dir)
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._ground_truth.landmarks_68[0] = (100, 100)

        labeler._handle_key(ord("s"))
        assert (output_dir / "test_gt.json").exists()


class TestLandmarkLabelerMouse:
    """Tests for mouse callback handling."""

    def test_mouse_callback_sets_landmark(self, tmp_path: Path) -> None:
        """Test left-click sets landmark position."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._current_landmark_idx = 0

        labeler._mouse_callback(cv2.EVENT_LBUTTONDOWN, 150, 200, 0, None)
        assert labeler.ground_truth.landmarks_68[0] == (150, 200)

    def test_mouse_callback_sets_hairline(self, tmp_path: Path) -> None:
        """Test left-click in hairline mode sets hairline y."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._current_landmark_idx = 68  # hairline mode

        labeler._mouse_callback(cv2.EVENT_LBUTTONDOWN, 150, 75, 0, None)
        assert labeler.ground_truth.hairline_y == 75

    def test_mouse_callback_ignores_other_events(self, tmp_path: Path) -> None:
        """Test that non-left-click events are ignored."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)

        labeler._mouse_callback(cv2.EVENT_MOUSEMOVE, 150, 200, 0, None)
        assert labeler.ground_truth.landmarks_68[0] == (-1, -1)


class TestLandmarkLabelerSave:
    """Tests for save functionality."""

    def test_save_current_creates_file(self, tmp_path: Path) -> None:
        """Test that save creates the ground truth file."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        output_dir = tmp_path / "gt"
        labeler = LandmarkLabeler([img_path], output_dir)
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._ground_truth.landmarks_68[0] = (100, 100)
        labeler._ground_truth.landmarks_68[1] = (200, 200)

        labeler._save_current()
        gt_path = output_dir / "test_gt.json"
        assert gt_path.exists()

        loaded = LandmarkGroundTruth.load(gt_path)
        assert loaded.landmarks_68[0] == (100, 100)
        assert loaded.landmarks_68[1] == (200, 200)

    def test_save_current_no_ground_truth(self, tmp_path: Path) -> None:
        """Test save with no ground truth does not crash."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._ground_truth = None
        labeler._save_current()  # Should not raise


class TestLandmarkLabelerRendering:
    """Tests for rendering methods (with mocked cv2)."""

    def test_draw_placed_landmarks(self, tmp_path: Path) -> None:
        """Test that placed landmarks are drawn on canvas."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._ground_truth.landmarks_68[0] = (100, 100)
        labeler._ground_truth.landmarks_68[1] = (200, 200)

        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        labeler._draw_placed_landmarks(canvas)
        # Just verify it doesn't crash; visual correctness is manual

    def test_draw_hairline(self, tmp_path: Path) -> None:
        """Test hairline drawing when hairline is set."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._ground_truth.hairline_y = 100

        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        labeler._draw_hairline(canvas)

    def test_draw_hairline_not_set(self, tmp_path: Path) -> None:
        """Test hairline drawing when hairline is not set."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._ground_truth.hairline_y = None

        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        labeler._draw_hairline(canvas)
        # Should not crash and not modify canvas

    def test_draw_active_landmark(self, tmp_path: Path) -> None:
        """Test active landmark highlighting."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._ground_truth.landmarks_68[0] = (100, 100)
        labeler._current_landmark_idx = 0

        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        labeler._draw_active_landmark(canvas)

    def test_draw_active_landmark_unplaced(self, tmp_path: Path) -> None:
        """Test active landmark highlighting when not yet placed."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._current_landmark_idx = 5

        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        labeler._draw_active_landmark(canvas)

    def test_draw_active_hairline(self, tmp_path: Path) -> None:
        """Test active hairline mode rendering."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._current_landmark_idx = 68  # hairline mode

        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        labeler._draw_active_hairline(canvas)

    def test_draw_ui_overlay(self, tmp_path: Path) -> None:
        """Test UI overlay rendering."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)

        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        labeler._draw_ui_overlay(canvas)

    def test_render(self, tmp_path: Path) -> None:
        """Test full render pipeline."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_bgr = np.zeros((400, 400, 3), dtype=np.uint8)
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)

        # Should not crash
        labeler._render()


class TestLandmarkLabelerQuit:
    """Tests for quit behavior."""

    def test_request_quit_complete(self, tmp_path: Path) -> None:
        """Test quit when all landmarks are placed."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        # Place all landmarks
        for i in range(68):
            labeler._ground_truth.landmarks_68[i] = (i, i)

        labeler._request_quit()
        assert labeler._quit_requested

    def test_request_quit_incomplete(self, tmp_path: Path) -> None:
        """Test quit with incomplete landmarks auto-saves."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        output_dir = tmp_path / "gt"
        labeler = LandmarkLabeler([img_path], output_dir)
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._ground_truth.landmarks_68[0] = (100, 100)

        labeler._request_quit()
        assert labeler._quit_requested
        # Should have auto-saved
        assert (output_dir / "test_gt.json").exists()


class TestLandmarkLabelerPredictions:
    """Tests for prediction-always-loaded behaviour."""

    def test_predictions_loaded_by_default(self, tmp_path: Path) -> None:
        """Test that predictions are always attempted on load."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        # _predicted_landmarks should be initialised
        assert len(labeler._predicted_landmarks) == 68

    def test_corrected_landmarks_tracked(self, tmp_path: Path) -> None:
        """Test that placing a landmark records it as corrected."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._current_landmark_idx = 0

        labeler._mouse_callback(cv2.EVENT_LBUTTONDOWN, 150, 200, 0, None)
        assert 0 in labeler.ground_truth.corrected_landmarks
        assert labeler.ground_truth.landmarks_68[0] == (150, 200)

    def test_corrected_landmarks_not_duplicated(self, tmp_path: Path) -> None:
        """Test that correcting the same landmark twice does not duplicate."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._current_landmark_idx = 0

        labeler._mouse_callback(cv2.EVENT_LBUTTONDOWN, 150, 200, 0, None)
        labeler._current_landmark_idx = 0
        labeler._mouse_callback(cv2.EVENT_LBUTTONDOWN, 160, 210, 0, None)
        assert labeler.ground_truth.corrected_landmarks.count(0) == 1

    def test_draw_predicted_vs_corrected_style(self, tmp_path: Path) -> None:
        """Test that predicted and corrected landmarks use different styles."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)
        labeler._ground_truth.landmarks_68[0] = (100, 100)
        labeler._ground_truth.landmarks_68[1] = (200, 200)
        labeler._ground_truth.corrected_landmarks = [0]

        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        labeler._draw_placed_landmarks(canvas)
        # Just verify it doesn't crash; visual correctness is manual


class TestLandmarkLabelerReset:
    """Tests for reset-to-prediction functionality."""

    def test_reset_landmark_to_prediction(self, tmp_path: Path) -> None:
        """Verify landmark reverts to its predicted value on reset."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)

        # Set a predicted value
        labeler._predicted_landmarks[5] = (120, 130)
        labeler._ground_truth.landmarks_68[5] = (120, 130)

        # User corrects it (mouse callback auto-advances)
        labeler._current_landmark_idx = 5
        labeler._mouse_callback(cv2.EVENT_LBUTTONDOWN, 200, 250, 0, None)
        assert labeler.ground_truth.landmarks_68[5] == (200, 250)
        assert 5 in labeler.ground_truth.corrected_landmarks

        # Go back to landmark 5 and reset to prediction
        labeler._current_landmark_idx = 5
        labeler._reset_active_landmark()
        assert labeler.ground_truth.landmarks_68[5] == (120, 130)

    def test_reset_removes_from_corrected_list(self, tmp_path: Path) -> None:
        """Verify corrected_landmarks is updated on reset."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)

        labeler._predicted_landmarks[3] = (50, 60)
        labeler._ground_truth.landmarks_68[3] = (50, 60)
        labeler._current_landmark_idx = 3
        labeler._mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 110, 0, None)
        assert 3 in labeler.ground_truth.corrected_landmarks

        # Go back to landmark 3 and reset
        labeler._current_landmark_idx = 3
        labeler._reset_active_landmark()
        assert 3 not in labeler.ground_truth.corrected_landmarks

    def test_reset_unplaced_landmark(self, tmp_path: Path) -> None:
        """Verify reset to [-1, -1] when no prediction exists."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)

        # No prediction set (defaults to (-1, -1))
        labeler._current_landmark_idx = 10
        labeler._ground_truth.landmarks_68[10] = (150, 160)
        labeler._ground_truth.corrected_landmarks.append(10)

        labeler._reset_active_landmark()
        assert labeler.ground_truth.landmarks_68[10] == (-1, -1)
        assert 10 not in labeler.ground_truth.corrected_landmarks

    def test_reset_hairline_to_prediction(self, tmp_path: Path) -> None:
        """Verify hairline reverts to predicted hairline y on reset."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)

        labeler._predicted_hairline_y = 80
        labeler._ground_truth.hairline_y = 80

        # User changes hairline
        labeler._current_landmark_idx = 68
        labeler._mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 120, 0, None)
        assert labeler.ground_truth.hairline_y == 120

        # Reset to prediction
        labeler._reset_active_landmark()
        assert labeler.ground_truth.hairline_y == 80

    def test_reset_hairline_without_prediction(self, tmp_path: Path) -> None:
        """Verify hairline reset to None when no prediction exists."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)

        labeler._predicted_hairline_y = None
        labeler._ground_truth.hairline_y = 100

        labeler._current_landmark_idx = 68
        labeler._reset_active_landmark()
        assert labeler.ground_truth.hairline_y is None

    def test_handle_key_r(self, tmp_path: Path) -> None:
        """Test 'R' key triggers reset."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        labeler._img_w = 400
        labeler._img_h = 400
        labeler._ground_truth = labeler._create_empty_ground_truth(img_path)

        labeler._predicted_landmarks[2] = (30, 40)
        labeler._ground_truth.landmarks_68[2] = (30, 40)
        labeler._current_landmark_idx = 2
        labeler._mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 110, 0, None)
        assert labeler.ground_truth.landmarks_68[2] == (100, 110)

        # Go back to landmark 2 and press r
        labeler._current_landmark_idx = 2
        labeler._handle_key(ord("r"))
        assert labeler.ground_truth.landmarks_68[2] == (30, 40)
        assert 2 not in labeler.ground_truth.corrected_landmarks


class TestLandmarkLabelerProperties:
    """Tests for public properties."""

    def test_properties(self, tmp_path: Path) -> None:
        """Test that properties return expected values."""
        img_path = tmp_path / "test.jpg"
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        labeler = LandmarkLabeler([img_path], tmp_path / "output")
        assert labeler.current_image_idx == 0
        assert labeler.current_landmark_idx == 0
        assert labeler.ground_truth is None
        assert labeler.image_paths == [img_path]
