"""Interactive OpenCV GUI for manually annotating 68 facial landmarks + hairline.

Mode 1 of the Landmark Evaluation Tool. Provides an interactive labeling
interface where users can click to place landmarks, navigate with keyboard
shortcuts, and save ground truth data to JSON.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from visagism.constants import (
    LABELER_ACTIVE_COLOR,
    LABELER_ACTIVE_RADIUS,
    LABELER_BG_COLOR,
    LABELER_FONT_SCALE,
    LABELER_FONT_THICKNESS,
    LABELER_HAIRLINE_DASH_LENGTH,
    LABELER_HAIRLINE_GAP_LENGTH,
    LABELER_HAIRLINE_THICKNESS,
    LABELER_INDEX_FONT_SCALE,
    LABELER_INSTRUCTION_FONT_SCALE,
    LABELER_INSTRUCTION_LINE_HEIGHT,
    LABELER_INSTRUCTION_MARGIN,
    LABELER_PLACED_RADIUS,
    LABELER_PREDICTED_COLOR,
    LABELER_PREDICTED_RADIUS,
    LABELER_TEXT_COLOR,
    LABELER_WINDOW_NAME,
    REGION_COLORS,
    REGION_LABELS,
    SUPPORTED_FORMATS,
)
from visagism.errors import LabelerError
from visagism.image_loader import ImageLoader
from visagism.landmark_ground_truth import LandmarkGroundTruth
from visagism.types import ImageArray


class LandmarkLabeler:
    """Interactive GUI for manual facial landmark annotation.

    Predictions from dlib are always loaded as starting points. Users
    see predicted landmarks in a muted style and correct them by
    clicking; corrected landmarks switch to the normal style.

    Parameters
    ----------
    image_paths : list of Path
        List of image files to label.
    output_dir : Path
        Directory where ground truth JSON files will be saved.
    """

    def __init__(
        self,
        image_paths: list[Path],
        output_dir: Path,
    ) -> None:
        """Initialise the labeler with a list of images."""
        self._image_paths = [
            p for p in image_paths
            if p.suffix.lower() in SUPPORTED_FORMATS
        ]
        if not self._image_paths:
            raise LabelerError("No valid image files found.")

        self._output_dir = output_dir

        # State
        self._current_image_idx: int = 0
        self._current_landmark_idx: int = 0  # 0-67 for landmarks, 68 for hairline
        self._img_bgr: ImageArray | None = None
        self._img_h: int = 0
        self._img_w: int = 0
        self._ground_truth: LandmarkGroundTruth | None = None
        self._window_created: bool = False
        self._quit_requested: bool = False

        # Predicted landmarks from dlib (always loaded)
        self._predicted_landmarks: list[tuple[int, int]] = [(-1, -1)] * 68
        self._predicted_hairline_y: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the interactive labeling loop.

        Creates an OpenCV window, loads the first image, and enters
        the keyboard/mouse event loop. Saves ground truth on exit.
        """
        self._create_window()
        self._load_image(0)

        while not self._quit_requested:
            self._render()
            key = cv2.waitKey(50) & 0xFF
            if key != 255:
                self._handle_key(key)

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Window / rendering
    # ------------------------------------------------------------------

    def _create_window(self) -> None:
        """Create the OpenCV window and register mouse callback."""
        cv2.namedWindow(LABELER_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(
            LABELER_WINDOW_NAME, self._mouse_callback, param=None,
        )
        self._window_created = True

    def _render(self) -> None:
        """Draw the current image with landmarks and UI overlay."""
        if self._img_bgr is None or self._ground_truth is None:
            return

        canvas = self._img_bgr.copy()

        # Draw placed landmarks
        self._draw_placed_landmarks(canvas)

        # Draw hairline if set
        self._draw_hairline(canvas)

        # Draw active landmark / hairline
        if self._current_landmark_idx < 68:
            self._draw_active_landmark(canvas)
        else:
            self._draw_active_hairline(canvas)

        # Draw UI overlay (instructions, progress)
        self._draw_ui_overlay(canvas)

        cv2.imshow(LABELER_WINDOW_NAME, canvas)

    def _draw_placed_landmarks(self, canvas: np.ndarray) -> None:
        """Draw all landmarks, distinguishing predicted vs corrected."""
        if self._ground_truth is None:
            return

        for i in range(68):
            pt = self._ground_truth.landmarks_68[i]
            if pt == (-1, -1):
                continue

            is_corrected = i in self._ground_truth.corrected_landmarks
            region = self._ground_truth.get_region_for_landmark(i)

            if is_corrected:
                color = REGION_COLORS.get(region, (200, 200, 200))
                radius = LABELER_PLACED_RADIUS
            else:
                color = LABELER_PREDICTED_COLOR
                radius = LABELER_PREDICTED_RADIUS

            cv2.circle(
                canvas, pt, radius, color, -1,
            )
            # Draw index number
            cv2.putText(
                canvas, str(i), (pt[0] + 5, pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, LABELER_INDEX_FONT_SCALE,
                LABELER_TEXT_COLOR, 1,
            )

    def _draw_hairline(self, canvas: np.ndarray) -> None:
        """Draw the hairline as a horizontal dashed line if set."""
        if (
            self._ground_truth is None
            or self._ground_truth.hairline_y is None
        ):
            return

        y = self._ground_truth.hairline_y
        h, w = canvas.shape[:2]
        color = LABELER_ACTIVE_COLOR

        dash_gap = LABELER_HAIRLINE_DASH_LENGTH + LABELER_HAIRLINE_GAP_LENGTH
        for x in range(0, w, dash_gap):
            x_end = min(x + LABELER_HAIRLINE_DASH_LENGTH, w)
            cv2.line(
                canvas, (x, y), (x_end, y), color,
                LABELER_HAIRLINE_THICKNESS,
            )

    def _draw_active_landmark(self, canvas: np.ndarray) -> None:
        """Highlight the currently active landmark."""
        if self._ground_truth is None:
            return

        idx = self._current_landmark_idx
        pt = self._ground_truth.landmarks_68[idx]

        # If not placed, draw at center of screen as a guide
        if pt == (-1, -1):
            pt = (self._img_w // 2, self._img_h // 2)

        region = self._ground_truth.get_region_for_landmark(idx)
        region_label = REGION_LABELS.get(region, region)

        # Draw larger active circle
        cv2.circle(
            canvas, pt, LABELER_ACTIVE_RADIUS, LABELER_ACTIVE_COLOR, 2,
        )

        # Draw index and region name
        text = f"{idx}: {region_label}"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, LABELER_FONT_SCALE,
            LABELER_FONT_THICKNESS,
        )
        tx = pt[0] + 12
        ty = pt[1] - 12

        # Background rectangle for text
        cv2.rectangle(
            canvas, (tx - 2, ty - th - 2),
            (tx + tw + 2, ty + 2), LABELER_BG_COLOR, -1,
        )
        cv2.putText(
            canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
            LABELER_FONT_SCALE, LABELER_TEXT_COLOR, LABELER_FONT_THICKNESS,
        )

    def _draw_active_hairline(self, canvas: np.ndarray) -> None:
        """Highlight the hairline labeling mode."""
        h, w = canvas.shape[:2]
        y = self._ground_truth.hairline_y if self._ground_truth else None

        if y is None:
            y = h // 4  # Default guide position

        # Draw guide line
        color = LABELER_ACTIVE_COLOR
        dash_gap = LABELER_HAIRLINE_DASH_LENGTH + LABELER_HAIRLINE_GAP_LENGTH
        for x in range(0, w, dash_gap):
            x_end = min(x + LABELER_HAIRLINE_DASH_LENGTH, w)
            cv2.line(
                canvas, (x, y), (x_end, y), color,
                LABELER_HAIRLINE_THICKNESS,
            )

        # Label
        text = "Hairline: click to set y-position"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, LABELER_FONT_SCALE,
            LABELER_FONT_THICKNESS,
        )
        tx = 10
        ty = y - 10
        cv2.rectangle(
            canvas, (tx - 2, ty - th - 2),
            (tx + tw + 2, ty + 2), LABELER_BG_COLOR, -1,
        )
        cv2.putText(
            canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
            LABELER_FONT_SCALE, LABELER_TEXT_COLOR, LABELER_FONT_THICKNESS,
        )

    def _draw_ui_overlay(self, canvas: np.ndarray) -> None:
        """Draw instructions and progress information."""
        if self._ground_truth is None:
            return

        h, w = canvas.shape[:2]
        placed, total = self._ground_truth.completion_count()
        img_name = self._image_paths[self._current_image_idx].name

        img_idx = self._current_image_idx + 1
        num_imgs = len(self._image_paths)
        hl_status = "set" if self._ground_truth.hairline_y is not None else "not set"
        lines = [
            f"Image {img_idx}/{num_imgs}: {img_name}",
            f"Landmarks: {placed}/{total} placed",
            f"Hairline: {hl_status}",
            "",
            "Controls:",
            "  Left-click: set landmark / hairline",
            "  Arrow keys / n / p: next/prev landmark",
            "  0-9: jump to landmark",
            "  Shift+N / Shift+P: next/prev image",
            "  r: reset to prediction",
            "  s: save",
            "  q / Esc: quit",
        ]

        x = LABELER_INSTRUCTION_MARGIN
        y = h - (len(lines) * LABELER_INSTRUCTION_LINE_HEIGHT)
        y -= LABELER_INSTRUCTION_MARGIN

        for line in lines:
            if line == "":
                y += LABELER_INSTRUCTION_LINE_HEIGHT
                continue
            (tw, th), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX,
                LABELER_INSTRUCTION_FONT_SCALE, 1,
            )
            cv2.rectangle(
                canvas, (x - 2, y - th - 2),
                (x + tw + 4, y + 4), LABELER_BG_COLOR, -1,
            )
            cv2.putText(
                canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                LABELER_INSTRUCTION_FONT_SCALE, LABELER_TEXT_COLOR, 1,
            )
            y += LABELER_INSTRUCTION_LINE_HEIGHT

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    def _mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: object | None,
    ) -> None:
        """Handle mouse events in the OpenCV window."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self._ground_truth is None:
            return

        if self._current_landmark_idx < 68:
            # Place landmark
            idx = self._current_landmark_idx
            self._ground_truth.landmarks_68[idx] = (x, y)
            if idx not in self._ground_truth.corrected_landmarks:
                self._ground_truth.corrected_landmarks.append(idx)
            # Auto-advance to next landmark
            self._next_landmark()
        else:
            # Place hairline
            self._ground_truth.hairline_y = y

    # ------------------------------------------------------------------
    # Keyboard handling
    # ------------------------------------------------------------------

    def _handle_key(self, key: int) -> None:
        """Process a keyboard key press."""
        # Navigation: next/prev landmark
        if key in (ord("n"), 83):  # 'n' or right arrow
            self._next_landmark()
        elif key in (ord("p"), 81):  # 'p' or left arrow
            self._prev_landmark()
        elif key in (ord("N"),):  # Shift+N: next image
            self._next_image()
        elif key in (ord("P"),):  # Shift+P: prev image
            self._prev_image()
        elif key == 82:  # up arrow
            self._prev_landmark()
        elif key == 84:  # down arrow
            self._next_landmark()

        # Direct jump: 0-9
        elif ord("0") <= key <= ord("9"):
            target = key - ord("0")
            if target < 68:
                self._current_landmark_idx = target

        # Save
        elif key == ord("s"):
            self._save_current()

        # Reset active landmark to prediction
        elif key == ord("r"):
            self._reset_active_landmark()

        # Quit
        elif key in (ord("q"), 27):  # 'q' or Esc
            self._request_quit()

    def _next_landmark(self) -> None:
        """Advance to the next landmark or hairline."""
        self._current_landmark_idx = min(68, self._current_landmark_idx + 1)

    def _prev_landmark(self) -> None:
        """Go back to the previous landmark."""
        self._current_landmark_idx = max(0, self._current_landmark_idx - 1)

    def _next_image(self) -> None:
        """Advance to the next image, auto-saving current."""
        if self._current_image_idx < len(self._image_paths) - 1:
            self._save_current()
            self._current_image_idx += 1
            self._load_image(self._current_image_idx)

    def _prev_image(self) -> None:
        """Go back to the previous image, auto-saving current."""
        if self._current_image_idx > 0:
            self._save_current()
            self._current_image_idx -= 1
            self._load_image(self._current_image_idx)

    def _request_quit(self) -> None:
        """Handle quit request with incomplete check."""
        if self._ground_truth is None:
            self._quit_requested = True
            return

        placed, total = self._ground_truth.completion_count()
        if placed < total:
            # Prompt user — in GUI mode we can't do interactive console
            # easily while OpenCV window is open, so auto-save and warn
            print(
                f"Warning: Only {placed}/{total} landmarks placed. "
                "Auto-saving before quit."
            )
            self._save_current()
        self._quit_requested = True

    def _reset_active_landmark(self) -> None:
        """Reset the currently active landmark to its predicted value.

        For landmarks 0-67, reverts ``_landmarks_68[idx]`` to the
        corresponding entry in ``_predicted_landmarks``.  For the
        hairline (index 68), reverts ``_hairline_y`` to
        ``_predicted_hairline_y``.

        If no prediction exists (e.g. detection failed), the landmark
        is reset to ``(-1, -1)`` and the hairline to ``None``.

        The index is removed from ``corrected_landmarks`` so the point
        is rendered in the predicted (gray) style again.
        """
        if self._ground_truth is None:
            return

        idx = self._current_landmark_idx

        if idx < 68:
            predicted = self._predicted_landmarks[idx]
            self._ground_truth.landmarks_68[idx] = predicted
            if idx in self._ground_truth.corrected_landmarks:
                self._ground_truth.corrected_landmarks.remove(idx)
        else:
            # Hairline
            if self._predicted_hairline_y is not None:
                self._ground_truth.hairline_y = self._predicted_hairline_y
            else:
                self._ground_truth.hairline_y = None

    # ------------------------------------------------------------------
    # Image / ground truth management
    # ------------------------------------------------------------------

    def _load_image(self, idx: int) -> None:
        """Load an image and its ground truth (or create new).

        Always runs the detection pipeline to obtain initial predictions.
        If detection fails, falls back to manual mode with all landmarks
        at ``(-1, -1)``.
        """
        path = self._image_paths[idx]
        try:
            img_bgr, _ = ImageLoader.load(path)
        except Exception as exc:
            raise LabelerError(f"Failed to load image {path}: {exc}") from exc

        self._img_bgr = img_bgr
        self._img_h, self._img_w = img_bgr.shape[:2]

        # Try to load existing ground truth
        gt_path = self._ground_truth_path(path)
        if gt_path.exists():
            try:
                self._ground_truth = LandmarkGroundTruth.load(gt_path)
                print(f"Resumed existing ground truth: {gt_path}")
            except Exception as exc:
                print(f"Warning: Could not load existing ground truth: {exc}")
                self._ground_truth = self._create_empty_ground_truth(path)
        else:
            self._ground_truth = self._create_empty_ground_truth(path)

        # Always seed with dlib predictions when no existing ground truth
        if self._all_unplaced():
            self._run_detection_pipeline(path, img_bgr)

        self._current_landmark_idx = 0

    def _create_empty_ground_truth(self, path: Path) -> LandmarkGroundTruth:
        """Create a new empty ground truth for an image."""
        return LandmarkGroundTruth(
            image_path=path,
            image_width=self._img_w,
            image_height=self._img_h,
            landmarks_68=[(-1, -1)] * 68,
            hairline_y=None,
            corrected_landmarks=[],
        )

    def _all_unplaced(self) -> bool:
        """Check if all landmarks are unplaced."""
        if self._ground_truth is None:
            return True
        placed, _ = self._ground_truth.completion_count()
        return placed == 0

    def _run_detection_pipeline(self, path: Path, img_bgr: np.ndarray) -> None:
        """Run FaceDetector → LandmarkDetector → HairlineDetector.

        Stores predicted landmarks and hairline as starting points.
        If any step fails, prints a warning and leaves ground truth
        in manual mode.
        """
        try:
            from visagism.face_detector import FaceDetector
            from visagism.hairline_detector import HairlineDetector
            from visagism.landmark_detector import LandmarkDetector
            from visagism.model_finder import ModelFinder

            model_path = ModelFinder.find()
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            face_rect = FaceDetector().detect(img_gray, (self._img_h, self._img_w))
            landmarks = LandmarkDetector(model_path).detect(
                img_gray, face_rect, path,
            )

            if self._ground_truth is not None:
                self._ground_truth.landmarks_68 = landmarks.landmarks_68
                self._predicted_landmarks = list(landmarks.landmarks_68)
                print(
                    f"Loaded {len(landmarks.landmarks_68)} "
                    f"predicted landmarks from dlib."
                )

            # Hairline detection
            try:
                hairline_result = HairlineDetector().detect(img_gray, landmarks)
                hairline_y = hairline_result.get("hairline_y")
                if hairline_y is not None and self._ground_truth is not None:
                    self._ground_truth.hairline_y = int(hairline_y)
                    self._predicted_hairline_y = int(hairline_y)
                    print(f"Predicted hairline at y={hairline_y}.")
            except Exception as exc:
                print(f"Warning: Hairline detection failed: {exc}")
                self._predicted_hairline_y = None

        except Exception as exc:
            print(
                f"Warning: Could not load predictions for {path.name}: {exc}\n"
                f"  Falling back to manual labeling from scratch."
            )
            self._predicted_landmarks = [(-1, -1)] * 68
            self._predicted_hairline_y = None

    def _save_current(self) -> None:
        """Save the current image's ground truth to disk."""
        if self._ground_truth is None:
            return

        path = self._ground_truth_path(self._image_paths[self._current_image_idx])
        try:
            self._ground_truth.save(path)
            placed, total = self._ground_truth.completion_count()
            print(
                f"Saved ground truth: {path} "
                f"({placed}/{total} landmarks)"
            )
        except Exception as exc:
            print(f"Warning: Failed to save ground truth: {exc}")

    def _ground_truth_path(self, image_path: Path) -> Path:
        """Return the ground truth JSON path for an image."""
        return self._output_dir / f"{image_path.stem}_gt.json"

    # ------------------------------------------------------------------
    # Properties for testing / inspection
    # ------------------------------------------------------------------

    @property
    def current_image_idx(self) -> int:
        """Index of the currently displayed image."""
        return self._current_image_idx

    @property
    def current_landmark_idx(self) -> int:
        """Index of the currently active landmark (0-68)."""
        return self._current_landmark_idx

    @property
    def ground_truth(self) -> LandmarkGroundTruth | None:
        """Current ground truth data."""
        return self._ground_truth

    @property
    def image_paths(self) -> list[Path]:
        """List of image paths being labeled."""
        return self._image_paths
